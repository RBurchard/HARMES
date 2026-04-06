from ..training.train import le
from ..config import config
from ..preprocessing.load_and_process_raw_data import get_part_datas
from ..training.train import run_exp

from sklearn.metrics import f1_score, accuracy_score
import shelve
import pandas as pd
import numpy as np

def calc_metrics(all_preds, all_true, test_order):
    r = []
    for i, (p, t, s) in enumerate(zip(all_preds, all_true, test_order)):
        p = le.inverse_transform(p)
        t = le.inverse_transform(t)
        r.append([f1_score(t, p, average="macro"), f1_score(t, p, average="weighted"), accuracy_score(p, t), s.zfill(2)])
    return r, f1_score(np.concatenate(all_true), np.concatenate(all_preds), average="macro")



# In[139]:

def evaluate_models():
    metr_dfs = []

    for ws in config["window_sizes"]:
        part_datas = get_part_datas(ws)
        for sensor_config in config["sensor_configs"]:
            s_cfg_str = "+".join(sensor_config)
            key = s_cfg_str + ";" + str(ws)
            with shelve.open("cache") as shf:
                try:
                    res = shf[key]
                except:
                    res = run_exp(part_datas, sensor_config)
                    shf[key] = res
                metrics, f1_macro = calc_metrics(*res)
                metr_df = pd.DataFrame(metrics, columns=["F1-Score (macro)", "F1-score (weighted)", "Accuracy", "Participant"])
                metr_df["window size"] = ws
                metr_df["sensor_config"] = s_cfg_str
                metr_dfs.append(metr_df)
    full_results_df = pd.concat(metr_dfs)

    full_results_df["Participant | Window Size"] = "P" + full_results_df.Participant + " | " + full_results_df[
        "window size"].astype(str) + "s"
    full_results_df.to_csv("full_results_df.csv")
    return full_results_df
