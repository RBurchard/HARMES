import pickle

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, KFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.nn.functional as F
import shelve

from ..models.MultiModalModel import MultiModalModel
from ..config import config
import os

label_order = [
    "Null",
    "Floor cleaning",
    "Making tea",
    "Vacuum Cleaning",
    "Window cleaning",
    "Brushing teeth",
    "Washing dishes",
    "Cutting vegetables",
    "Putting away the dishes",
    "Cleaning out the dishwasher",
    "Watering plant",
    "Cleaning table",
    "Washing hands",
    "Drinking",
    "Cream hands",
    "Disinfecting hands"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

le = LabelEncoder()
le.fit(label_order)
hum_targets = le.transform(["Washing dishes", "Washing hands"])

sle = LabelEncoder()
samosa_labels = ['Alarm_clock', 'Blender_in_use', 'Brushing_hair', 'Chopping',
        'Clapping', 'Coughing', 'Drill in use', 'Drinking', 'Grating',
        'Hair_dryer_in_use', 'Hammering', 'Knocking', 'Laughing',
        'Microwave', 'Other', 'Pouring_pitcher', 'Sanding', 'Scratching',
        'Screwing', 'Shaver_in_use', 'Toilet_flushing', 'Toothbrushing',
        'Twisting_jar', 'Vacuum in use', 'Washing_Utensils',
        'Washing_hands', 'Wiping_with_rag']
sle.fit(samosa_labels)

def train_model(sensor_config, writer, Xw1_train, Xw2_train,
                Xwa_train, Xwh_train, y_train, n_classes=16, teacher_logits=None):
    y_train_np = y_train.cpu().numpy()
    counts = np.bincount(y_train_np, minlength=n_classes)

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    if writer is None:
        writer = SummaryWriter()


    ## sensor config:
    imu_l = "imu_l" in sensor_config or "all" in sensor_config
    imu_r = "imu_r" in sensor_config or "all" in sensor_config
    audio = "a" in sensor_config or "all" in sensor_config
    humid = "h" in sensor_config or "all" in sensor_config
    print(imu_l, imu_r, audio, humid, sensor_config)

    model = nn.DataParallel(MultiModalModel(n_classes, imu_l=imu_l, imu_r=imu_r, audio=audio, humid=humid)).to(
        device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epochs = config["epochs"]
    batch_size = config["batch_size"]


    # default parameters from implementaion of AudioIMU student teacher training
    alpha = 0.2
    Temp = 4

    for epoch in tqdm(range(epochs)):

        perm = torch.randperm(Xw2_train.size(0))

        loss_ttl = 0
        for i in range(0, Xw2_train.size(0), batch_size):

            idx = perm[i:i + batch_size]
            if Xw1_train is None and Xwh_train is None:
                xbw1 = None
                xbwh = None
            else:
                xbw1 = Xw1_train[idx].to(device)
                xbwh = Xwh_train[idx].to(device)
            xbw2 = Xw2_train[idx].to(device)
            xbwa = Xwa_train[idx].to(device)
            if teacher_logits is not None:
                t_logits = teacher_logits[idx].to(device)

            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            # We pass all sensor data to model.forward here, but it simply ignores the de-selected sensors
            out = model(audio=xbwa, imu_left=xbw1, imu_right=xbw2, humid=xbwh)
            if teacher_logits is None:
                loss = criterion(out, yb)
            else:
                loss = criterion(out, yb) * alpha + (1 - alpha) * kd_loss(out, t_logits, T=Temp, alpha=1)

            loss.backward()

            loss_ttl += loss.item() * len(idx)

            optimizer.step()
        writer.add_scalar('Loss/train', loss_ttl / Xw2_train.size(0), epoch)
    model.eval()
    return model


def eval_model(model, Xw1_test, Xw2_test, Xwa_test,
               Xwh_test, y_test, return_logits=False):
    batch_size = 64  # adjust based on your GPU, for testing - param does NOT influence results, just inference speed
    preds = []

    with torch.no_grad():
        for i in range(0, len(Xw1_test), batch_size):
            Xw1_batch = Xw1_test[i:i + batch_size].to(device)
            Xw2_batch = Xw2_test[i:i + batch_size].to(device)
            Xwa_batch = Xwa_test[i:i + batch_size].to(device)
            Xwh_batch = Xwh_test[i:i + batch_size].to(device)

            logits = model(Xwa_batch, Xw1_batch, Xw2_batch, Xwh_batch)
            pred = logits if return_logits is not False else torch.argmax(logits, dim=1)

            preds.append(pred.cpu())
        # concatenate all batches
        pred = torch.cat(preds).numpy()

    y_test = y_test.cpu().numpy()

    return pred, y_test





def run_exp(part_datas, sensor_config, model_save_str,
            pretrain=False, samosa=False, audio_ablation=False, envs=None, hum_only=False):
    torch.cuda.empty_cache()
    cv = LeaveOneOut()
    if envs is not None:
        cv = LeaveOneGroupOut()
    subs = np.array(list(part_datas.keys()))

    writer = SummaryWriter()

    all_preds = []
    all_true = []
    all_labels = []
    for sub in part_datas:
        if samosa:
            all_labels.extend(part_datas[sub][2])
        else:
            all_labels.extend(part_datas[sub][4])

    test_order = []

    if pretrain:  # train a single model on the entire dataset on the given sensors etc.
        try:
            state_dict = torch.load(model_save_str  + ("_sms_" if samosa else "") + f"_transfer_pretrained.pth")
            model = MultiModalModel(len(samosa_labels) if samosa else 16, imu_l=False, imu_r=True, audio=True, humid=False)
            model.load_state_dict(state_dict)
            model = nn.DataParallel(model).to(device)
        except:
            w1, w2, wa, wh, labels = [], [], [], [], []
            # ----- training data -----
            for sub in subs:
                if samosa:
                    w2.extend(part_datas[sub][0])
                    wa.extend(part_datas[sub][1])
                    labels.extend(part_datas[sub][2])
                else:
                    w1.extend(part_datas[sub][0])
                    w2.extend(part_datas[sub][1])
                    wa.extend(part_datas[sub][2])
                    wh.extend(part_datas[sub][3])
                    labels.extend(part_datas[sub][4])
            if samosa:
                y_train = sle.transform(labels)
                Xw1_train = None
                Xwh_train = None
            else:
                y_train = le.transform(labels)
                Xw1_train = torch.tensor(np.stack(w1), dtype=torch.float32)
                Xwh_train = torch.tensor(np.stack(wh), dtype=torch.float32)
            Xw2_train = torch.tensor(np.stack(w2), dtype=torch.float32)
            Xwa_train = torch.tensor(np.stack(wa), dtype=torch.float32)


            y_train = torch.tensor(y_train, dtype=torch.long)
            model = train_model(sensor_config, writer, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train,
                                n_classes=len(samosa_labels) if samosa else 16)
            torch.save(model.state_dict(), model_save_str  + ("_sms_" if samosa else "") + f"_transfer_pretrained.pth")
        return model

    elif audio_ablation:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        preds_both, preds_imu, y_trues = [], [], []
        for sub_no, (train_subs, test_subs) in enumerate(tqdm(list(cv.split(subs)))):
            w1, w2, wa, wh, labels = [], [], [], [], []
            # ----- training data -----
            for sub in tqdm(train_subs):
                w1.extend(part_datas[subs[sub]][0])
                w2.extend(part_datas[subs[sub]][1])
                wa.extend(part_datas[subs[sub]][2])
                wh.extend(part_datas[subs[sub]][3])
                labels.extend(part_datas[subs[sub]][4])

            y_train = le.transform(labels)
            Xw1_train = torch.tensor(np.stack(w1), dtype=torch.float32)
            Xw2_train = torch.tensor(np.stack(w2), dtype=torch.float32)
            Xwa_train = torch.tensor(np.stack(wa), dtype=torch.float32)
            Xwh_train = torch.tensor(np.stack(wh), dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)

            w1, w2, wa, wh, labels = [], [], [], [], []
            for sub in tqdm(test_subs):
                w1.extend(part_datas[subs[sub]][0])
                w2.extend(part_datas[subs[sub]][1])
                wa.extend(part_datas[subs[sub]][2])
                wh.extend(part_datas[subs[sub]][3])
                labels.extend(part_datas[subs[sub]][4])

            y_test = le.transform(labels)


            Xw1_test = torch.tensor(np.stack(w1), dtype=torch.float32)
            Xw2_test = torch.tensor(np.stack(w2), dtype=torch.float32)
            Xwa_test = torch.tensor(np.zeros_like(np.stack(wa)), dtype=torch.float32)  # zero all audio for test
            Xwh_test = torch.tensor(np.stack(wh), dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long).to(device)

            model_both = train_model(sensor_config, writer, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train)
            model_imu_r = train_model(["imu_r"], writer, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train)
            #torch.save(model.state_dict(), model_save_str + f"_part_{subs[test_sub[0]]}.pth")

            del Xw1_train, Xw2_train, Xwa_train, Xwh_train

            pred_both, y_test_both = eval_model(model_both, Xw1_test, Xw2_test, Xwa_test, Xwh_test, y_test)
            pred_imu, _ = eval_model(model_imu_r, Xw1_test, Xw2_test, Xwa_test, Xwh_test, y_test)

            preds_both.append(pred_both)
            y_trues.append(y_test_both)
            preds_imu.append(pred_imu)

            # force Python garbage collection
            # wait for GPU ops to finish
            torch.cuda.synchronize()
            # finally clear cache
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        return np.concatenate(preds_both), np.concatenate(preds_imu), np.concatenate(y_trues)

    ## else: if no pretraining or audio ablation: run the "normal" lopo evaluation

    for sub_no, (train_subs, test_sub) in enumerate(tqdm(list(cv.split(subs, groups=envs)))):

        w1, w2, wa, wh, labels = [], [], [], [], []

        if envs is None:
            test_order.append(subs[test_sub[0]])

        else:
            test_order.append([subs[idx] for idx in test_sub])

        # ----- training data -----
        for sub in tqdm(train_subs):
            w1.extend(part_datas[subs[sub]][0])
            w2.extend(part_datas[subs[sub]][1])
            wa.extend(part_datas[subs[sub]][2])
            wh.extend(part_datas[subs[sub]][3])
            labels.extend(part_datas[subs[sub]][4])

        y_train = le.transform(labels)
        if (hum_only):
            y_train = np.isin(y_train, hum_targets).astype(int)

        w1t, w2t, wat, wht, labelst = [], [], [], [], []

        for sub in test_sub:
            w1t.extend(part_datas[subs[sub]][0])
            w2t.extend(part_datas[subs[sub]][1])
            wat.extend(part_datas[subs[sub]][2])
            wht.extend(part_datas[subs[sub]][3])
            labelst.extend(part_datas[subs[sub]][4])


        Xw1_test = torch.tensor(np.stack(w1t), dtype=torch.float32)
        Xw2_test = torch.tensor(np.stack(w2t), dtype=torch.float32)
        Xwa_test = torch.tensor(np.stack(wat), dtype=torch.float32)
        Xwh_test = torch.tensor(np.stack(wht), dtype=torch.float32)

        y_test = le.transform(labelst)
        if (hum_only):
            y_test = np.isin(y_test, hum_targets).astype(int)


        Xw1_train = torch.tensor(np.stack(w1), dtype=torch.float32)
        Xw2_train = torch.tensor(np.stack(w2), dtype=torch.float32)
        Xwa_train = torch.tensor(np.stack(wa), dtype=torch.float32)
        Xwh_train = torch.tensor(np.stack(wh), dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        model = train_model(sensor_config, writer, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train,
                            n_classes=2 if hum_only else 16)
        if envs is None:
            torch.save(model.state_dict(), model_save_str + f"_part_{subs[test_sub[0]]}.pth")




        batch_size = 64  # adjust based on your GPU, for testing - param does NOT influence results, just inference speed

        preds = []
        del Xw1_train, Xw2_train, Xwa_train, Xwh_train
        with torch.no_grad():
            for i in range(0, len(Xw1_test), batch_size):
                Xw1_batch = Xw1_test[i:i+batch_size].to(device)
                Xw2_batch = Xw2_test[i:i+batch_size].to(device)
                Xwa_batch = Xwa_test[i:i+batch_size].to(device)
                Xwh_batch = Xwh_test[i:i+batch_size].to(device)

                logits = model(Xwa_batch, Xw1_batch, Xw2_batch, Xwh_batch)
                pred = torch.argmax(logits, dim=1)

                preds.append(pred.cpu())
            # concatenate all batches
            pred = torch.cat(preds).numpy()

        y_test = y_test.cpu().numpy()

        acc = np.mean(pred == y_test)
        writer.add_scalar('Acc/test', acc, sub_no)

        writer.close()
        all_preds.append(pred)
        all_true.append(y_test)

        # delete everything GPU-related
        del model
        del y_train, y_test
        del Xw1_test, Xw2_test, Xwa_test, Xwh_test

        # force Python garbage collection
        import gc
        gc.collect()

        # wait for GPU ops to finish
        torch.cuda.synchronize()

        # finally clear cache
        torch.cuda.empty_cache()


    # ----- aggregated metrics -----
    print("Aggregated results across all LOSO folds:\n")
    pred_labels = le.inverse_transform(np.concatenate(all_preds))
    true_labels = le.inverse_transform(np.concatenate(all_true))
    print(classification_report(true_labels, pred_labels))

    return all_preds, all_true, test_order


def run_transfer(model, participant_datas,
                 frozen=False, evaluate_scratch=True, linear_probing=False,
                 samosa=True, repetition=0):

    loo = LeaveOneOut()
    subs = list(participant_datas.keys())
    if samosa:
        n_classes = len(samosa_labels)
    else:
        n_classes = len(label_order)
    sle = LabelEncoder()
    if samosa:
        sle.fit(samosa_labels)
    else:
        sle.fit(label_order)

    scratch = [], []
    pre_trained = [], []

    saved_state = model.module.state_dict()

    key = f"transfer_ckpt_{frozen}_{evaluate_scratch}_{linear_probing}_{samosa}_rep_{repetition}"

    # ----- training data -----
    for train_subs, test_subs in loo.split(subs):
        split_key = key + f"_{str(test_subs)}"
        with shelve.open("cache_dir/ckpt") as shf:
            try:
                pred, pred_scratch, y_test = shf[split_key]
                pre_trained[0].extend(pred)
                pre_trained[1].extend(y_test)
                scratch[0].extend(pred_scratch)
                scratch[1].extend(y_test)
                continue
            except:
                pass



        w_imu, w_a, labels = [], [], []
        for sub in train_subs:
            if samosa:
                w_imu.extend(participant_datas[subs[sub]][0])
                w_a.extend(participant_datas[subs[sub]][1])
                labels.extend(participant_datas[subs[sub]][2])
            else:
                w_imu.extend(participant_datas[subs[sub]][1])
                w_a.extend(participant_datas[subs[sub]][2])
                labels.extend(participant_datas[subs[sub]][4])

        if samosa:
            w_imu_test, w_a_test = [torch.tensor(x, dtype=torch.float32) for x in participant_datas[subs[test_subs[0]]][:2]]
            y_test = sle.transform(participant_datas[subs[test_subs[0]]][2])
        else:
            w_imu_test, w_a_test = [torch.tensor(x, dtype=torch.float32) for x in
                                    participant_datas[subs[test_subs[0]]][1:3]]
            y_test = sle.transform(participant_datas[subs[test_subs[0]]][4])

        y_train = sle.transform(labels)

        Xw1_train = torch.tensor(np.stack(w_imu), dtype=torch.float32)
        Xwa_train = torch.tensor(np.stack(w_a), dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        ## train and evaluate for each participant:
        # 1) train and evaluate a model from scratch.
        if evaluate_scratch:
            new_model = nn.DataParallel(
                MultiModalModel(
                    n_classes,
                    imu_l=False,
                    imu_r=True,
                    audio=True,
                    humid=False
                )
            ).to(device)

            if linear_probing:  # replaces the usual classification head with a single linear layer
                new_model.module.enable_linear_probing(n_classes)
            train_samosa(new_model, Xw1_train, Xwa_train, y_train)
            pred_scratch = evaluate_samosa(new_model, Xw1_test=w_imu_test, Xwa_test=w_a_test, y_test=y_test)
            scratch[0].extend(pred_scratch)
            scratch[1].extend(list(y_test.cpu()))

        # 2) train and evaluate the pre-trained model (and remember freeze its weights / reset its output size).
        if samosa:
            n_classes_model = 16  # reset to correct shape for loading pre-trained weights
        else:
            n_classes_model = len(samosa_labels)

        model = nn.DataParallel(
            MultiModalModel(
                n_classes_model,
                imu_l=False,
                imu_r=True,
                audio=True,
                humid=False
            )
        ).to(device)

        model.module.load_state_dict(saved_state)  # reset to same pre-trained model every time.
        if frozen:
            model.module.freeze_encoders(freeze=True)
        model.module.reset_classification_head(n_classes)
        if linear_probing:   # replaces the usual classification head
            model.module.enable_linear_probing(n_classes)
        model = model.to(device)

        train_samosa(model, Xw1_train, Xwa_train, y_train)
        pred = evaluate_samosa(model, Xw1_test=w_imu_test, Xwa_test=w_a_test, y_test=y_test)
        pre_trained[0].extend(pred)
        pre_trained[1].extend(list(y_test.cpu()))

        with shelve.open("cache_dir/ckpt") as shf:
            shf[split_key] = pred, pred_scratch, list(y_test.cpu())

    return pre_trained, scratch



def train_samosa(model, Xw1_train, Xwa_train, y_train):
    model.train()
    model.to(device)
    y_train_np = y_train.cpu().numpy()
    counts = np.bincount(y_train_np, minlength=len(le.classes_))

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epochs = 10  # config["epochs"]
    batch_size = config["batch_size"]

    for i in range(torch.cuda.device_count()):
        try:
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"GPU {i}: ERROR: {e}")

    print("=" * 50)

    for epoch in tqdm(range(epochs)):

        perm = torch.randperm(Xw1_train.size(0))

        loss_ttl = 0
        for i in range(0, Xw1_train.size(0), batch_size):
            idx = perm[i:i + batch_size]

            xbw1 = None
            xbw2 = Xw1_train[idx].to(device)  # Pass as right wrist data like in Samosa paper.
            xbwa = Xwa_train[idx].to(device)
            xbwh = None
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            # We pass all sensor data to model.forward here, but it simply ignores the de-selected sensors
            out = model(audio=xbwa, imu_left=xbw1, imu_right=xbw2, humid=xbwh)

            loss = criterion(out, yb)

            loss.backward()

            loss_ttl += loss.item() * len(idx)

            optimizer.step()
    model.eval()


def evaluate_samosa(model, Xw1_test, Xwa_test, y_test):
    batch_size = 64
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xw1_test), batch_size):
            Xw2_batch = Xw1_test[i:i + batch_size].to(device)
            Xw1_batch = None
            Xwa_batch = Xwa_test[i:i + batch_size].to(device)
            Xwh_batch = None

            logits = model(Xwa_batch, Xw1_batch, Xw2_batch, Xwh_batch)
            pred = torch.argmax(logits, dim=1)

            preds.append(pred.cpu())
        # concatenate all batches
        pred = torch.cat(preds).numpy()
    y_test = y_test.cpu().numpy()
    print("----" * 15)
    print(f"Evaluating samosa...: ACC: {accuracy_score(y_test, pred)}, F1: {f1_score(y_test, pred, average='macro')}")
    print("(final results averaged macro across all participant's predictions)")
    return pred


# kd_loss function taken from AudioIMU: Enhancing Inertial Sensing-Based Activity Recognition with Acoustic Models
# https://github.com/Human-Signals-Lab/AudioIMU/blob/main/joint_trainfixlr_loso_individual.py
def kd_loss(logits, truth, T=8.0, alpha=0.9):
    # print(T)
    # print(y)

    # truth =F.one_hot(truth, num_classes=23)
    p = F.log_softmax(logits/T, dim=1)
    q = F.softmax(truth/T, dim=1)
    # print(logits.size(), truth.size(), p.size(), q.size())
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / logits.shape[0]
    # print(l_kl * alpha)
    return l_kl * alpha


def run_teacher_student(part_datas, model_save_str):
    cv = LeaveOneOut()
    subs = np.array(list(part_datas.keys()))
    preds_both, preds_imu, preds_imu_transfer, y_trues = [], [], [], []
    for sub_no, (train_subs, test_subs) in enumerate(tqdm(list(cv.split(subs)))):
        w1, w2, wa, wh, labels = [], [], [], [], []
        # ----- training data -----
        for sub in tqdm(train_subs):
            w1.extend(part_datas[subs[sub]][0])
            w2.extend(part_datas[subs[sub]][1])
            wa.extend(part_datas[subs[sub]][2])
            wh.extend(part_datas[subs[sub]][3])
            labels.extend(part_datas[subs[sub]][4])

        y_train = le.transform(labels)
        Xw1_train = torch.tensor(np.stack(w1), dtype=torch.float32)
        Xw2_train = torch.tensor(np.stack(w2), dtype=torch.float32)
        Xwa_train = torch.tensor(np.stack(wa), dtype=torch.float32)
        Xwh_train = torch.tensor(np.stack(wh), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        w1, w2, wa, wh, labels = [], [], [], [], []
        for sub in tqdm(test_subs):
            w1.extend(part_datas[subs[sub]][0])
            w2.extend(part_datas[subs[sub]][1])
            wa.extend(part_datas[subs[sub]][2])
            wh.extend(part_datas[subs[sub]][3])
            labels.extend(part_datas[subs[sub]][4])

        y_test = le.transform(labels)

        Xw1_test = torch.tensor(np.stack(w1), dtype=torch.float32)
        Xw2_test = torch.tensor(np.stack(w2), dtype=torch.float32)
        Xwa_test = torch.tensor(np.stack(wa), dtype=torch.float32)
        Xwh_test = torch.tensor(np.stack(wh), dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        # trains one multimodal model
        model_both = train_model(["a", "imu_r"], None, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train)
        # plus one imu only model
        model_imu_r = train_model(["imu_r"], None, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train)
        # plus one with datatransfer, i.e., put in predictions from model_both as teacher_logits:
        teacher_logits, _ = eval_model(model_both, Xw1_train, Xw2_train, Xwa_train, Xwh_train, y_train, return_logits=True)
        teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32)
        model_transfer = train_model(["imu_r"], None, Xw1_train, Xw2_train, Xwa_train,
                                     Xwh_train, y_train, teacher_logits=teacher_logits)

        # torch.save(model.state_dict(), model_save_str + f"_part_{subs[test_sub[0]]}.pth")

        del Xw1_train, Xw2_train, Xwa_train, Xwh_train

        pred_both, y_test_both = eval_model(model_both, Xw1_test, Xw2_test, Xwa_test, Xwh_test, y_test)
        pred_imu, _ = eval_model(model_imu_r, Xw1_test, Xw2_test, Xwa_test, Xwh_test, y_test)
        pred_imu_transfer, _ = eval_model(model_transfer, Xw1_test, Xw2_test, Xwa_test, Xwh_test, y_test)

        preds_both.append(pred_both)
        y_trues.append(y_test_both)
        preds_imu.append(pred_imu)
        preds_imu_transfer.append(pred_imu_transfer)

        # force Python garbage collection
        # wait for GPU ops to finish
        torch.cuda.synchronize()
        # finally clear cache
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return np.concatenate(preds_both), np.concatenate(preds_imu), np.concatenate(preds_imu_transfer), np.concatenate(y_trues)
