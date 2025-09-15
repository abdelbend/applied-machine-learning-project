

import numpy as np
import pandas as pd
from scipy.signal import detrend, windows
from scipy.stats import iqr


def find_dominant_frequencies(x: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with the fast fourier transformation.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequencies for each signal, shape: (num_samples,).
    """

    if x.ndim == 1:
        x = x.reshape(1, -1)

    num_samples, seq_len = x.shape
    dominant_freqs = np.full(num_samples, np.nan)  # Initialize with NaN

    if seq_len == 0:
        # zero-length sequence
        return dominant_freqs

    # Apply a window
    window = windows.get_window("hann", seq_len)

    for i in range(num_samples):
        signal = x[i, :]

        # Detrend the signal to remove linear trends
        detrended_signal = detrend(signal)

        # Apply window
        windowed_signal = detrended_signal * window

        # Compute FFT for real input
        yf = np.fft.rfft(windowed_signal)

        # xf contains the corresponding frequencies
        xf = np.fft.rfftfreq(seq_len, 1 / fs)

        psd = np.abs(yf) ** 2

        idx_max_amplitude = np.argmax(psd[1:]) + 1
        dominant_freqs[i] = xf[idx_max_amplitude]

    return dominant_freqs


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extract 20 different features from the data.
    Args:
        data (np.ndarray): The data to extract features from.
        labels (np.ndarray): The labels of the data.
    Returns:
        pd.DataFrame: The extracted features.
    """
    n_samples = data.shape[0]

    current_signals = data[:, :, 0]
    voltage_signals = data[:, :, 1]
    power_signals = current_signals * voltage_signals

    features_list = []

    features_list.append(
        pd.Series(np.mean(current_signals, axis=1), name='current_mean'))
    features_list.append(
        pd.Series(np.std(current_signals, axis=1), name='current_std'))
    features_list.append(
        pd.Series(np.median(current_signals, axis=1), name='current_median'))
    features_list.append(
        pd.Series(np.sqrt(np.mean(current_signals**2, axis=1)), name='current_rms'))
    features_list.append(
        pd.Series(np.min(current_signals, axis=1), name='current_min'))
    features_list.append(
        pd.Series(np.max(current_signals, axis=1), name='current_max'))
    features_list.append(pd.Series(
        iqr(current_signals, axis=1, nan_policy='propagate'), name='current_iqr'))
    features_list.append(pd.Series(find_dominant_frequencies(
        current_signals, fs=1.0), name='current_dom_freq'))

    features_list.append(
        pd.Series(np.mean(voltage_signals, axis=1), name='voltage_mean'))
    features_list.append(
        pd.Series(np.std(voltage_signals, axis=1), name='voltage_std'))
    features_list.append(
        pd.Series(np.median(voltage_signals, axis=1), name='voltage_median'))
    features_list.append(
        pd.Series(np.sqrt(np.mean(voltage_signals**2, axis=1)), name='voltage_rms'))
    features_list.append(
        pd.Series(np.min(voltage_signals, axis=1), name='voltage_min'))
    features_list.append(
        pd.Series(np.max(voltage_signals, axis=1), name='voltage_max'))
    features_list.append(pd.Series(
        iqr(voltage_signals, axis=1, nan_policy='propagate'), name='voltage_iqr'))
    features_list.append(pd.Series(find_dominant_frequencies(
        voltage_signals, fs=1.0), name='voltage_dom_freq'))

    features_list.append(
        pd.Series(np.mean(power_signals, axis=1), name='power_mean'))
    features_list.append(
        pd.Series(np.std(power_signals, axis=1), name='power_std'))
    features_list.append(
        pd.Series(np.sqrt(np.mean(power_signals**2, axis=1)), name='power_rms'))
    features_list.append(
        pd.Series(np.max(power_signals, axis=1), name='power_max'))

    features_df = pd.concat(features_list, axis=1)

    labels_1d = labels.ravel()

    features_df['label'] = labels_1d

    return features_df
