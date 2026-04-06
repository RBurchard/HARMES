import librosa
import numpy as np
import pandas as pd







def extract_log_mel_spectrogram_sliding_window(audio_data, sr=44100, window_size=2, hop_size=2, n_mels=128, n_fft=2048, hop_length=None, labels_df=None, normalize=False):
    """
    Computes log-Mel spectrogram features using a sliding window approach.

    Parameters:
        - audio_data: 1D numpy array of raw audio
        - sr: Sampling rate
        - window_size: Duration of each window in seconds
        - hop_size: Hop size between windows in seconds
        - n_mels: Number of Mel bands
        - n_fft: FFT window size
        - hop_length: STFT hop length (in samples); defaults to 1/4 of n_fft

    Returns:
        - A 3D numpy array of shape (num_windows, n_mels, num_frames_per_window)
    """
    win_length = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    hop_length = hop_length or n_fft // 4
    num_windows = (len(audio_data) - win_length) // hop_samples + 1

    segments = []

    for i in range(num_windows):
        start = i * hop_samples
        end = start + win_length
        y_win = audio_data[start:end]

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y_win,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0  # Power spectrogram (amplitude squared)
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        segments.append(log_mel_spec)


    labels = np.zeros(len(segments))

    if labels_df is not None:
        for _, line in labels_df.iterrows():
            if (line.stop_sample - line.start_sample) / sr < 2:
                continue
            start = (line.start_sample - win_length + hop_samples) // hop_samples
            stop = (line.stop_sample - win_length + hop_samples) // hop_samples
            labels[start:stop] = 1

    segments = np.array(segments)


    return segments, labels # Shape: (num_windows, n_mels, frames_per_window)

def humid_features(humid):
    humid = np.asarray(humid)

    # Basic stats
    mean = np.mean(humid)
    std = np.std(humid)
    max_val = np.max(humid)
    min_val = np.min(humid)
    median = np.median(humid)

    # Range & robust dispersion
    value_range = np.ptp(humid)  # max - min
    iqr = np.percentile(humid, 75) - np.percentile(humid, 25)

    # Dynamics
    diff = np.diff(humid)
    mean_diff = np.mean(diff) if len(diff) > 0 else 0
    std_diff = np.std(diff) if len(diff) > 0 else 0
    max_abs_diff = np.max(np.abs(diff)) if len(diff) > 0 else 0

    # Trend (linear slope)
    if len(humid) > 1:
        slope = np.polyfit(np.arange(len(humid)), humid, 1)[0]
    else:
        slope = 0

    # Threshold-based features (example threshold = 50)
    frac_above_50 = np.mean(humid > 50)
    crossings_50 = np.sum(np.diff(humid > 50) != 0)

    # Distribution shape
    #skewness = skew(humid) if len(humid) > 2 else 0
    #kurt = kurtosis(humid) if len(humid) > 3 else 0

    # Energy
    energy = np.sum(humid ** 2)

    return np.array([
        mean,
        std,
        max_val,
        min_val,
        median,
        value_range,
        iqr,
        mean_diff,
        std_diff,
        max_abs_diff,
        slope,
        frac_above_50,
        crossings_50,
        #skewness,
        #kurt,
        energy
    ])



def majority_label(intervals, start_time, end_time):

    overlap = (
        intervals["end"].clip(upper=end_time) -
        intervals["start"].clip(lower=start_time)
    ).clip(lower=0)


    idx = overlap.idxmax()

    if overlap.loc[idx] == 0:
        return "Null"

    return intervals.loc[idx, "label"]


humid_feature_dim = len(humid_features(np.zeros(500)))