from ..training.train import le, run_transfer, run_teacher_student
from ..config import config
from ..preprocessing.load_and_process_raw_data import get_part_datas, load_samosa
from ..training.train import run_exp

from sklearn.metrics import f1_score, accuracy_score
import shelve
import pandas as pd
import numpy as np
import sys
from time import time


def calc_metrics(all_preds, all_true, test_order):
    r = []
    for i, (p, t, s) in enumerate(zip(all_preds, all_true, test_order)):
        p = le.inverse_transform(p)
        t = le.inverse_transform(t)
        r.append([f1_score(t, p, average="macro"), f1_score(t, p, average="weighted"), accuracy_score(p, t), s.zfill(2)])
    return r, f1_score(np.concatenate(all_true), np.concatenate(all_preds), average="macro")


def evaluate_models():
    metr_dfs = []
    start_time = time()
    time_limit = 24 * 60 * 60 - 10 * 60  # around 24 h
    run_time = 0
    for ws in config["window_sizes"]:
        part_datas = get_part_datas(ws)
        for sensor_config in config["sensor_configs"]:
            s_cfg_str = "+".join(sensor_config)
            for repetition in range(config["experiment_repetitions"]):
                key = s_cfg_str + ";" + str(ws)+ ";" + f"rep_{repetition}"
                with shelve.open("cache") as shf:
                    try:
                        res = shf[key]
                        if res == "running":
                            continue
                    except:
                        res = "do_run"
                if res == "do_run":
                    if time() - start_time + run_time * 1.05 > time_limit:
                        print("next run wouldn't finish before time limit, exiting.")
                        sys.exit(0)
                    with shelve.open("cache") as shf:
                        shf[key] = "running"
                    save_str = f"runs/model_weights/{str(ws).zfill(2)}_{'-'.join(sensor_config)}"
                    run_start = time()
                    res = run_exp(part_datas, sensor_config, save_str, pretrain=False)  # runs lopo evaluation
                    run_time = time() - run_start
                with shelve.open("cache") as shf:
                    shf[key] = res
                metrics, f1_macro = calc_metrics(*res)
                metr_df = pd.DataFrame(metrics, columns=["F1-Score (macro)", "F1-score (weighted)", "Accuracy", "Participant"])
                metr_df["window size"] = ws
                metr_df["sensor_config"] = s_cfg_str
                metr_df["repetition"] = repetition
                metr_dfs.append(metr_df)
    full_results_df = pd.concat(metr_dfs)

    full_results_df["Participant | Window Size"] = "P" + full_results_df.Participant + " | " + full_results_df[
        "window size"].astype(str) + "s"
    full_results_df.to_csv("full_results_df.csv")
    return full_results_df


def evaluate_transfer():
    ws = 2  # like samosa IMU, although it has ~3s audio windows.
    part_datas = get_part_datas(ws, audio_sr=1000)  # 1000 Hz like in samosa experiments
    sensor_config = ["imu_r", "a"]
    s_cfg_str = "+".join(sensor_config) + "_transfer"
    base_key = s_cfg_str + ";" + str(ws)

    for repetition in range(5):
        key = base_key + f";rep_{repetition}"
        with shelve.open("cache") as shf:
            try:
                val = shf[key]
                if val == "running":
                    continue
                res, res_scratch, res_harm, res_harm_scratch = val
                run = False
            except:
                save_str = f"runs/model_weights/{str(ws).zfill(2)}_{'-'.join(sensor_config)}_transfer"
                run = True
                shf[key] = "running"
        if run:
            print("running transfer learning rep:", repetition)
            try:
                with shelve.open("cache") as shf:
                    model = shf[key + "_pretrained_model"]
            except:
                model = run_exp(part_datas, sensor_config, save_str, pretrain=True)
                with shelve.open("cache") as shf:
                    shf[key + "_pretrained_model"] = model

            samosa_datas = load_samosa(ws=ws)

            try:
                with shelve.open("cache") as shf:
                    model_sms = shf[key + "_sms_pretrained_model"]
            except:
                model_sms = run_exp(samosa_datas, sensor_config, save_str + "_sms", pretrain=True, samosa=True)
                with shelve.open("cache") as shf:
                    shf[key + "_sms_pretrained_model"] = model_sms

            print("pre-training models completed")

            #### train and evaluate on lopo with frozen / unfrozen weights
            # unfrozen:
            res, res_scratch = run_transfer(model, samosa_datas, False, evaluate_scratch=True, linear_probing=False, repetition=repetition)
            #res_frozen = run_samosa(model, samosa_datas, True, evaluate_scratch=False)

            res_harm, res_harm_scratch = run_transfer(model_sms, part_datas, False, evaluate_scratch=True,
                                                          linear_probing=False, samosa=False, repetition=repetition)
            with shelve.open("cache") as shf:
                shf[key] = res, res_scratch, res_harm, res_harm_scratch
        f1_macro_h_to_s = f1_score(res[1], res[0], average="macro")
        f1_macro_s_scratch = f1_score(res_scratch[1], res_scratch[0], average="macro")
        f1_macro_s_to_h = f1_score(res_harm[1], res_harm[0], average="macro")
        f1_macro_h_scratch = f1_score(res_harm_scratch[1], res_harm_scratch[0], average="macro")
        #if len (res_scratch[0]) > 0:
        #    f1_macro_scratch = f1_score(res_scratch[1], res_scratch[0], average="macro")
        print("final results: harmes pre -> samosa", f1_macro_h_to_s)
        print("final results: samosa from scratch:", f1_macro_s_scratch)
        print("final results: samosa pre -> harmes", f1_macro_s_to_h)
        print("final results: harmes from scratch", f1_macro_h_scratch)


def evaluate_modality_transfer():
    """
    runs a 3-fold CV modality transfer experiment:
    does it help to train with IMU AND Audio data, when evaluating on IMU data only?
    - with finetuning
    - with no finetuning
    """

    ws = 5
    part_datas = get_part_datas(ws)
    sensor_config = ["imu_r", "a"]
    s_cfg_str = "+".join(sensor_config) + "_modality_transfer"
    for rep in range(5):
        key = s_cfg_str + ";" + str(ws) + f"_rep{rep}"
        run = False
        with shelve.open("cache") as shf:
            try:
                if shf[key] == "running":
                    print("rep:", rep+1, "already running. skipping...")
                    continue
                else:
                    preds_both, preds_imu, preds_imu_transfer, y_true = shf[key]
            except:
                run = True
        if run:
            with shelve.open("cache") as shf:
                shf[key] = "running"
            save_str = f"runs/model_weights/{str(ws).zfill(2)}_{'-'.join(sensor_config)}_modality_transfer_rep_{rep}"
            preds_both, preds_imu, preds_imu_transfer, y_true = run_teacher_student(part_datas, save_str)
            with shelve.open("cache") as shf:
                shf[key] = preds_both, preds_imu, preds_imu_transfer, y_true
        f1_macro_both = f1_score(y_true, preds_both, average="macro")
        f1_macro_imu = f1_score(y_true, preds_imu, average="macro")
        f1_macro_imu_transfer = f1_score(y_true, preds_imu_transfer, average="macro")

        print("final results: trained on audio and imu, evaluated on Both", f1_macro_both)
        print("final results: trained on IMU, evaluated on IMU:", f1_macro_imu)
        print("final results: trained on IMU + audio logits, evaluated on IMU:", f1_macro_imu_transfer)
        # metrics, f1_macro = calc_metrics(*res)

            # metr_df = pd.DataFrame(metrics, columns=["F1-Score (macro)", "F1-score (weighted)", "Accuracy", "Participant"])
            # metr_df["window size"] = ws
            # metr_df["sensor_config"] = s_cfg_str
            # metr_df.append(metr_df)
    # todo: fix below
    # full_results_df = pd.concat(metr_df)
    # full_results_df["Participant | Window Size"] = "P" + full_results_df.Participant + " | " + full_results_df[
    #    "window size"].astype(str) + "s"
    # full_results_df.to_csv("full_results_df.csv")
    # return full_results_df

def evaluate_envs():
    ws = 5
    part_datas = get_part_datas(ws)
    sensor_config = ["imu_r", "imu_l", "a"]
    s_cfg_str = "+".join(sensor_config) + "environment_test"
    key = s_cfg_str + ";" + str(ws)
    table = """01	1
    02	1
    03	2
    04	3
    05	4
    06	4
    07	4
    08	4
    09	5
    10	5
    11	6
    12	3
    13	7
    14	2
    15	8
    16	2
    17	2
    18	2
    19	4
    20	4"""
    part_env_dict = dict([s.split("\t") for s in table.split("\n")])
    envs_list = list(part_env_dict.values())

    try:
        with shelve.open("cache") as shf:
            res = shf[key]
    except:
        save_str = f"runs/model_weights/environment_test"
        res = run_exp(part_datas, sensor_config, save_str, pretrain=False, envs=envs_list)  # runs lopo evaluation
        with shelve.open("cache") as shf:
            shf[key] = res

def evaluate_humid_only():
    ws = 30
    part_datas = get_part_datas(ws)
    for repetition in range(5):
        for sensor_config in [["h"], ["all"], ["imu_r", "imu_l", "a"], ["imu_r"], ["imu_r", "h"]]:
            s_cfg_str = "+".join(sensor_config) + "humidity_only"
            key = s_cfg_str + ";" + str(ws) + f"rep_{repetition}"
            try:
                with shelve.open("cache") as shf:
                    res = shf[key]
                    if res == "running":
                        continue
            except:
                with shelve.open("cache") as shf:
                    shf[key] = "running"
                save_str = f"runs/model_weights/environment_test"
                res = run_exp(part_datas, sensor_config, save_str, pretrain=False, hum_only=True)  # runs lopo evaluation
                with shelve.open("cache") as shf:
                    shf[key] = res
