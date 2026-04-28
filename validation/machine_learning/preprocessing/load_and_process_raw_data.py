from ..config import base_path
from .extract_all_features import *

from glob import glob
import h5py
from sklearn.preprocessing import StandardScaler
import shelve
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from copy import copy
import pickle

feature_cols = ["acc_" + x for x in "xyz"] + ["gyro_" + x for x in "xyz"]


def get_audio(rec_path):
    x = glob(rec_path + "/*H5.h5")[0]
    return h5py.File(x)

def get_labels(rec_path):
    return glob(rec_path + "/*[0-9][0-9][0-9][0-9][0-9].csv")[0]

def get_puck(rec_path):
    return glob(rec_path + "/*_merged.csv")[0]

def get_watch(rec_path):
    return glob(rec_path + "/recording*_[0-9][0-9].csv")[0]


# In[7]:


def get_participants(base_path):
    return glob(base_path + "/*")

def get_recordings(part_path):
    return glob(part_path + "/*")


def load_one_recording(recording):
    scaler = StandardScaler()
    label_df = pd.read_csv(get_labels(recording))
    label_df = clean_label_df(label_df)

    df = pd.read_csv(get_puck(recording))
    df["time"] = pd.to_timedelta(df["ts_sync"], unit="ms")
    df = df.set_index("time")
    df = (
        df.select_dtypes(include="number")
        .resample("20ms")
        .mean()
        .interpolate()
    )
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df2 = pd.read_csv(get_watch(recording))
    df2["time"] = pd.to_timedelta(df2["ts_sync"], unit="ms")
    df2 = df2.set_index("time")
    df2 = (
        df2.select_dtypes(include="number")
        .resample("20ms")
        .mean()
        .interpolate()
    )
    df2[feature_cols] = scaler.fit_transform(df2[feature_cols])

    af = get_audio(recording)
    ad = af["audio"]
    ad = np.array(ad).astype(np.float32) / 32768.0

    return df, df2, ad, label_df


def clean_label_df(label_df):
    label_df["ts_sync"] = (label_df.Time - label_df.Time.iloc[0]) * 1000
    label_df = label_df[~label_df["Description"].str.contains("deleted", na=False)]
    label_df = label_df.iloc[1:]
    return label_df


def handle_split_recordings(rec1, rec2):
    label_df1 = pd.read_csv(get_labels(rec1))
    label_df1 = clean_label_df(label_df1)
    label_df2 = pd.read_csv(get_labels(rec2))
    task_list = pd.read_csv(glob(rec1 + "/*TASKLIST.csv")[0])

    # Find the checkpoint, at which the split happened, by looking at the first 4 entries of the second recording's labels
    seq = label_df2[label_df2.Type == "end"].Description.to_list()[:4]
    tl = task_list.TASKS.to_list()
    ind = [i for i in range(len(tl)) if tl[i:i + len(seq)] == seq][0]
    # The last recording should end with these 5 (which were conducted before the checkpoint)
    end_seq = tl[ind - 5:ind - 1]
    labels_1 = label_df1[label_df1.Type == "end"].Description.to_list()
    indices = label_df1[label_df1.Type == "end"].index
    # Match the sequence to an index
    ind = [i + len(end_seq) - 1 for i in range(len(labels_1)) if labels_1[i:i + len(end_seq)] == end_seq][0]

    # calculate timestamp for end of first recording.
    last_line1 = label_df1.loc[indices[ind]]
    last_ts = last_line1.ts_sync + 1000

    df1, df2, ad, label_df = load_one_recording(rec1)
    df1 = df1[df1.ts_sync < last_ts]
    df2 = df2[df2.ts_sync < last_ts]
    ad = ad[:int(44100 * last_ts / 1000)]
    return (df1, df2, ad, label_df), load_one_recording(rec2)



def run_export():
    # this will load all recordings into memory and store the result in a pkl file
    all_data = dict()
    print("Exporting all data to pickle file: dataset.pkl.xz")
    print("basepath", base_path)
    # columns to use for the pickle export:
    timestamp_col = "ts_sync"
    imu_cols = ["acc_" + d for d in "xyz"] + ["gyro_" + d for d in "xyz"]
    bme_cols = ["press", "temp", "humid"]
    imu_df_cols = imu_cols + [timestamp_col]
    bme_df_cols = bme_cols + [timestamp_col]
    for j, part_path in enumerate(tqdm(sorted(get_participants(base_path)))):
        part_rec_list = []
        part = part_path[-2:]
        recordings = get_recordings(part_path)
        datas = []
        labels = []
        for recording in recordings:
            rec = recording[-4:]
            try:
                df, df2, ad, label_df = load_one_recording(recording)
                rec_dict = {"IMU_L" : df[imu_df_cols], "IMU_R": df2[imu_df_cols],
                        "Audio" : ad, "BME280" : df[bme_df_cols], "Labels": label_df, "rec_id" : rec}
                part_rec_list.append(rec_dict)
            except Exception as e:
                rec1 = recording + "/" + recording[-4:] + "01"
                rec2 = recording + "/" + recording[-4:] + "02"
                print("recordings: ", rec1, rec2)
                r1, r2 = handle_split_recordings(rec1, rec2)
                for i, (df, df2, ad, label_df) in enumerate([r1, r2]):
                    rec_dict = {"IMU_L" : df[imu_df_cols], "IMU_R": df2[imu_df_cols],
                            "Audio" : ad, "BME280" : df[bme_df_cols], "Labels": label_df, "rec_id" : rec + ["01", "02"][i]}
                    part_rec_list.append(rec_dict)
         
                print("split data combined successfully")
        all_data[part] = part_rec_list
    with open("AudioDS/dataset.pkl", "wb") as f:
        pickle.dump(all_data, f)
        print("Saved in dataset.pkl")

def get_part_datas(ws=10):
    with shelve.open("cache") as shf:
        key = f"part_datas_{ws}"
        try:
            part_datas = shf[key]
            print("loaded from shelf")
        except:
            print("loading from shelf failed, reloading from disk")
            part_datas = {}



            for j, part_path in enumerate(tqdm(sorted(get_participants(base_path)))):
                part = part_path[-2:]
                recordings = get_recordings(part_path)
                datas = []
                labels = []
                for recording in recordings:
                    rec = recording[-4:]
                    if rec[-1] == "4":
                        continue  # don't load last recording for now.
                    try:
                        df, df2, ad, label_df = load_one_recording(recording)
                        cur_data, lab = window_data_single(part + "_" + rec[-2:], df, df2, label_df, ad, ws, ws)
                        datas.append(cur_data)
                        labels.append(lab)
                    except Exception as e:
                        print(f"recording: {recording}:", e)
                        rec1 = recording + "/" + recording[-4:] + "01"
                        rec2 = recording + "/" + recording[-4:] + "02"
                        print("recordings: ", rec1, rec2)
                        r1, r2 = handle_split_recordings(rec1, rec2)
                        for df, df2, ad, label_df in [r1, r2]:
                            print(df.shape, df2.shape, len(ad), label_df.shape)
                            cur_data, lab = window_data_single(part + "_" + rec[-2:], df, df2, label_df, ad, ws, ws)
                            datas.append(cur_data)
                            labels.append(lab)
                        print("split data combined successfully")
                for d, l in tqdm(zip(datas, labels), total=len(labels)):
                    sub_d = part_datas.get(part, [[], [], [], [], []])
                    new = [extract_log_mel_spectrogram_sliding_window(x, 44100, ws, ws)[0][0] for x in d[2] if
                           len(x) == 44100 * ws]
                    humid = [humid_features(x) for x in d[3] if len(x) == 50 * ws]
                    min_len = min((len(new), len(d[0]), len(d[1]), len(d[2]), len(d[3]), len(d[1])))
                    sub_d[0].extend(d[0][:min_len])
                    sub_d[1].extend(d[1][:min_len])
                    sub_d[2].extend(new[:min_len])
                    if len(new) != len(d[1]):
                        print(part, len(new), len(d[1]))
                    sub_d[3].extend(humid[:min_len])
                    sub_d[4].extend(l[:min_len])  # labels
                    part_datas[part] = sub_d

                for part in part_datas.keys():
                    sub_a = np.array(part_datas[part][2])
                    mean = sub_a.mean()
                    std = sub_a.std()
                    sub_a = (sub_a - mean) / std
                    part_datas[part][2] = sub_a

                    sub_h = np.array(part_datas[part][3])
                    mean = sub_h.mean(axis=0)
                    std = sub_h.std(axis=0)
                    if std.mean() == 0:
                        sub_h[:] = 0
                    else:
                        sub_h = (sub_h - mean) / std
                    part_datas[part][3] = sub_h
            print("writing data to shelf... (this might take a while)")
            shf[key] = part_datas
            print("finished writing data to shelf.")
    return part_datas


def window_data(puck_dfs, watch_dfs, label_dfs, audios, order, window_size=10, window_shift=10):
    data = []
    order_iter = copy(order)
    for part_rec, puck, watch, audio, labels in zip(order_iter, puck_dfs, watch_dfs, audios, label_dfs):
        r = window_data_single(part_rec, puck, watch, labels, audio, window_size, window_shift)
        if r is None:
            order.remove(part_rec)
            continue
        else:
            cur_data, lab = r
        data.append([cur_data, lab])
    return data, order


def window_data_single(part_rec, puck, watch, labels, audio, window_size=10, window_shift=10):
    cur_data = [[], [], [], []]
    lab = []

    starts = labels[labels["Type"] == "start"].copy()
    stops  = labels[labels["Type"] == "end"].copy()
    print(part_rec, len(starts), len(stops))
    try:
        intervals = pd.DataFrame({
        "label": starts["Description"].values,
        "start": starts["ts_sync"].values,
        "end":  stops["ts_sync"].values
    })
    except:
        print(part_rec, "error")
        return None

    min_len = min([int(puck.ts_sync.iloc[-1] / 1000), int(watch.ts_sync.iloc[-1] / 1000)])

    for start_time in range(0, min_len-window_size, window_shift):
        start_time = start_time * 1000
        end_time = start_time + window_size * 1000

        p = puck[(puck.ts_sync >= start_time)&(puck.ts_sync< end_time)][["acc_" + x for x in "xyz"] + ["gyro_" + x for x in "xyz"]].to_numpy()
        w = watch[(watch.ts_sync >= start_time )&(watch.ts_sync< end_time)][["acc_" + x for x in "xyz"]  + ["gyro_" + x for x in "xyz"]].to_numpy()
        l = majority_label(intervals, start_time, end_time)
        a = audio[int(start_time * 44.1):int(end_time * 44.1)]
        h = puck[(puck.ts_sync >= start_time)&(puck.ts_sync< end_time)]["humid"].to_numpy()

        assert len(p) == len(w) == len(h)

        cur_data[0].append(p)
        cur_data[1].append(w)
        cur_data[2].append(a)
        cur_data[3].append(h)
        lab.append(l)
    return cur_data, lab
