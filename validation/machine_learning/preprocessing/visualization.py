for j, part_path in enumerate(tqdm(get_participants(base_path))):
    break  # Skip generating visualizations
    recordings = get_recordings(part_path)
    pp = PdfPages(f"viz/viz_{j + 2}.pdf")
    for recording in recordings:
        try:
            print(f"recording: {recording}")
            label_df = pd.read_csv(get_labels(recording))
            label_df["ts_sync"] = (label_df.Time - label_df.Time.iloc[0]) * 1000

            all_labels = label_df.Description.iloc[1:].unique()
            af = get_audio(recording)
            ad = af["audio"]
            ad = np.array(ad).astype(np.float32) / 32768.0

            df = pd.read_csv(get_puck(recording))
            df2 = pd.read_csv(get_watch(recording))

            for label in all_labels:
                l_df = label_df[label_df.Description == label]
                l_start = l_df[l_df.Type == "start"]
                l_stop = l_df[l_df.Type == "end"]

                for i, (start, stop) in enumerate(zip(l_start.ts_sync.to_list(), l_stop.ts_sync.to_list())):
                    length = int((stop - start) / 1000)
                    start_samples = int(start * 44.1)
                    stop_samples = int(stop * 44.1)
                    spt = extract_log_mel_spectrogram_sliding_window(ad[start_samples:stop_samples], sr=44100,
                                                                     window_size=length, hop_size=length)[0][0]
                    fig, axes = plt.subplots(4, 1, figsize=(8, 10), )  # gridspec_kw={'width_ratios': [2, 1]}
                    axes[0].imshow(spt, aspect="auto")

                    axes[1].plot(df[(df.ts_sync > start) & (df.ts_sync < stop)][["acc_" + x for x in "xyz"]])
                    axes[1].set_title("left hand")
                    axes[1].set_xlim(
                        df[(df.ts_sync > start) & (df.ts_sync < stop)][["acc_" + x for x in "xyz"]].index[0],
                        df[(df.ts_sync > start) & (df.ts_sync < stop)][["acc_" + x for x in "xyz"]].index[-1])
                    axes[2].plot(df2[(df2.ts_sync > start) & (df2.ts_sync < stop)][["acc_" + x for x in "xyz"]])
                    axes[2].set_title("right hand")
                    axes[2].set_xlim(
                        df2[(df2.ts_sync > start) & (df2.ts_sync < stop)][["acc_" + x for x in "xyz"]].index[0],
                        df2[(df2.ts_sync > start) & (df2.ts_sync < stop)][["acc_" + x for x in "xyz"]].index[-1])
                    axes[3].plot(df[(df.ts_sync > start) & (df.ts_sync < stop)]["humid"])
                    axes[3].set_title("humidity")
                    axes[3].set_xlim(df[(df.ts_sync > start) & (df.ts_sync < stop)]["humid"].index[0],
                                     df[(df.ts_sync > start) & (df.ts_sync < stop)]["humid"].index[-1])
                    plt.suptitle(f"{recording}: {label} {i + 1}:")
                    plt.tight_layout()

                    pp.savefig(fig)
                    plt.close()
        except Exception as e:
            print(f"recording: {recording} failed:", e)
    pp.close()
# In[ ]: