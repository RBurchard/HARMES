import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..models.MultiModalModel import MultiModalModel
from ..config import config

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


le = LabelEncoder()
le.fit(label_order)


def run_exp(part_datas, sensor_config):
    torch.cuda.empty_cache()
    loo = LeaveOneOut()
    subs = np.array(list(part_datas.keys()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_true = []
    all_labels = []
    for sub in part_datas:
        all_labels.extend(part_datas[sub][4])

    test_order = []

    for sub_no, (train_subs, test_sub) in enumerate(tqdm(list(loo.split(subs)))):

        w1, w2, wa, wh, labels = [], [], [], [], []

        test_order.append(subs[test_sub[0]])

        # ----- training data -----
        for sub in tqdm(train_subs):
            w1.extend(part_datas[subs[sub]][0])
            w2.extend(part_datas[subs[sub]][1])
            wa.extend(part_datas[subs[sub]][2])
            wh.extend(part_datas[subs[sub]][3])
            labels.extend(part_datas[subs[sub]][4])



        y_train = np.array(labels)
        y_train = le.transform(labels)


        Xw1_test = torch.tensor(part_datas[subs[test_sub[0]]][0], dtype=torch.float32)
        Xw2_test = torch.tensor(part_datas[subs[test_sub[0]]][1], dtype=torch.float32)
        Xwa_test = torch.tensor(part_datas[subs[test_sub[0]]][2], dtype=torch.float32)
        Xwh_test = torch.tensor(part_datas[subs[test_sub[0]]][3], dtype=torch.float32)
        y_test = le.transform(part_datas[subs[test_sub[0]]][4])


        Xw1_train = torch.tensor(np.stack(w1), dtype=torch.float32)
        Xw2_train = torch.tensor(np.stack(w2), dtype=torch.float32)
   
        Xwa_train = torch.tensor(np.stack(wa), dtype=torch.float32)
        Xwh_train = torch.tensor(np.stack(wh), dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)

        y_train_np = y_train.cpu().numpy()
        counts = np.bincount(y_train_np, minlength=len(le.classes_))

        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)

        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        n_classes = len(le.classes_)
        ## sensor config:
        imu_l = "imu_l" in sensor_config or "all" in sensor_config
        imu_r = "imu_r" in sensor_config or "all" in sensor_config
        audio = "a" in sensor_config or "all" in sensor_config
        humid = "h" in sensor_config or "all" in sensor_config
        print(imu_l, imu_r, audio, humid, sensor_config)


        model = nn.DataParallel(MultiModalModel(n_classes, imu_l=imu_l, imu_r=imu_r, audio=audio, humid=humid)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)


        epochs = config["epochs"]
        batch_size = config["batch_size"]

        writer = SummaryWriter()
        for epoch in tqdm(range(epochs)):

            perm = torch.randperm(Xw1_train.size(0))

            loss_ttl = 0
            for i in range(0, Xw1_train.size(0), batch_size):

                idx = perm[i:i+batch_size]

                xbw1 = Xw1_train[idx].to(device)
                xbw2 = Xw2_train[idx].to(device)
                xbwa = Xwa_train[idx].to(device)
                xbwh = Xwh_train[idx].to(device)
                yb = y_train[idx].to(device)

                optimizer.zero_grad()
                # We pass all sensor data to model.forward here, but it simply ignores the de-selected sensors
                out = model(audio=xbwa, imu_left=xbw1, imu_right=xbw2, humid=xbwh)

                loss = criterion(out, yb)

                loss.backward()

                loss_ttl += loss.item() * len(idx)

                optimizer.step()
            writer.add_scalar('Loss/train', loss_ttl / Xw1_train.size(0), epoch)
        model.eval()


        batch_size = 64  # adjust based on your GPU

        preds = []
        del Xw1_train, Xw2_train, Xwa_train, Xwh_train,  xbw1, xbw2, xbwa, xbwh, out
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
        optimizer.state.clear()

        # delete everything GPU-related
        del model, optimizer, criterion
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
