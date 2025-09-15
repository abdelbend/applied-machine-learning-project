import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pathlib import Path
import re


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"No file in Path {data_path} is found")

    df = pd.read_csv(data_path)

    df = remove_unlabeled_data(df)
    try:
        for cols in df.columns:
            df[cols] = df[cols].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError(
            "umeric values detected in numeric timeseries columns.")

    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            "Dataframe empty after removing unlabeled data and dropping NaN values")

    return df


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """
    df = data[data["labels"] != -1].copy()
    return df


def col_num(col: str) -> int:
    """
    Returns number of the cycle
    """
    m = re.search(r'(\d+)$', col)
    return int(m.group(1))


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.
    """
    labels = data['labels'].to_numpy()
    exp_ids = data['exp_ids'].to_numpy()

    # Get I and V columns, then sort them numerically to avoid I1, I10, I2 sorting issues
    curr_cols = sorted(
        [col for col in data.columns if col.startswith(
            "I") and col not in ['exp_ids', 'labels']],
        key=col_num
    )
    vol_cols = sorted(
        [col for col in data.columns if col.startswith("V")],
        key=col_num
    )

    currents = data[curr_cols].to_numpy()
    voltages = data[vol_cols].to_numpy()

    # Stack into shape [n_samples, timesteps, 2]
    curr_and_volts = np.stack((currents, voltages), axis=-1)

    return labels, exp_ids, curr_and_volts


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.

    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window

    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """

    # Input shape: (n_samples, timesteps, features)
    # shape: (n_windows, timesteps, features, sequence_length)
    view = sliding_window_view(data, window_shape=(sequence_length,), axis=0)

    # Target shape: (n_windows, sequence_length, timesteps, features)
    view_permuted = np.moveaxis(view, -1, 1)

    # shape (n_windows, sequence_length * timesteps, features)
    n_windows = view_permuted.shape[0]
    n_timesteps = data.shape[1]
    n_features = data.shape[2]

    reshaped_view = view_permuted.reshape(
        n_windows, sequence_length * n_timesteps, n_features)

    return reshaped_view


def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    pathpre = path.with_suffix("")
    cache_feat = pathpre.with_name(f"{pathpre.name}_features.npy")
    cache_labels = pathpre.with_name(f"{pathpre.name}_labels.npy")
    cache_exp = pathpre.with_name(f"{pathpre.name}_exp_ids.npy")

    # check is cache esxists
    if cache_feat.exists() and cache_labels.exists() and cache_exp.exists():
        features = np.load(cache_feat)
        labels = np.load(cache_labels)
        exp_ids = np.load(cache_exp)
    else:
        df = load_data(path)
        labels, exp_ids, features = convert_to_np(df)

        np.save(cache_feat, features)
        np.save(cache_labels, labels)
        np.save(cache_exp, exp_ids)

    if return_sequences:
        features = create_sliding_windows_first_dim(features, sequence_length)

        # match *(n_windows, sequence_length)*
        labels = np.lib.stride_tricks.sliding_window_view(
            labels, window_shape=sequence_length
        )
        exp_ids = np.lib.stride_tricks.sliding_window_view(
            exp_ids, window_shape=sequence_length
        )

    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if n_samples > len(labels):
            raise ValueError(
                f"Requested n_samples={n_samples}, but only {len(labels)} available")
        rng = np.random.default_rng()
        idx = rng.choice(len(labels), size=n_samples, replace=False)

        features = features[idx]
        labels = labels[idx]
        exp_ids = exp_ids[idx]

    return features, labels, exp_ids
