from typing import Literal
import numpy as np
import pandas as pd
import logging  # For potential logging, if needed
from tqdm import tqdm  # For progress bars
from dtaidistance import dtw  # For Dynamic Time Warping

# Define type hints for clarity
DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]


class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.

    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean",
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """

    def __init__(self,
                 k: int,
                 max_iter: int = 100,
                 distance_metric: DISTANCE_METRICS = "euclidean",
                 init_method: INIT_METHOD = "kmeans++"
                 ):
        self.k: int = k
        self.max_iter: int = max_iter
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.centroids: np.ndarray | None = None
        self.inertia_: float | None = None
        self.labels_: np.ndarray | None = None  # Cluster labels for each point
        self.init_method: INIT_METHOD = init_method

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids based on the chosen init_method.

        Args:
            x (np.ndarray): Training data of shape (n_samples, n_features) or (n_samples, time_steps, n_features_ts).

        Returns:
            np.ndarray: Initialized centroids.
        """
        n_samples = x.shape[0]

        if self.init_method == "random":
            rng = np.random.default_rng(42)
            # Select k unique random samples as initial centroids
            random_indices = self.rng.choice(
                n_samples, size=self.k, replace=False)
            centroids = x[random_indices].copy()

        elif self.init_method == "kmeans++":
            # (k, n_features) or (k, time_steps, n_features_ts)
            centroids = np.empty((self.k,) + x.shape[1:], dtype=x.dtype)

            # 1. Choose the first centroid randomly from the data points
            rng = np.random.default_rng(42)

            # Select k unique random samples as initial centroids
            first_centroid_idx = rng.choice(n_samples)
            centroids[0] = x[first_centroid_idx].copy()

            # For the remaining k-1 centroids
            for i in range(1, self.k):

                # For each data point, find the squared distance to the nearest *already chosen* centroid
                distances_to_chosen_centroids = self._compute_distance(
                    x, centroids[:i])
                min_squared_distances = np.min(
                    distances_to_chosen_centroids**2, axis=1)

                # 2b. Select the next centroid with probability proportional to D(x)^2
                # Avoid division by zero if all points are identical to centroids
                if np.sum(min_squared_distances) == 0:
                    # Fallback to random sampling if all distances are zero
                    probabilities = np.ones(n_samples) / n_samples
                else:
                    probabilities = min_squared_distances / \
                        np.sum(min_squared_distances)

                rng = np.random.default_rng(42)
                next_centroid_idx = rng.choice(
                    n_samples, p=probabilities)
                centroids[i] = x[next_centroid_idx].copy()
        else:
            raise ValueError(
                f"Unknown initialization method: {self.init_method}")

        return centroids

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each sample in x and each centroid.

        Args:
            x (np.ndarray): Data points. Shape (n_samples, n_features) for 2D data,
                            or (n_samples, time_steps, n_features_ts) for 3D time series data.
            centroids (np.ndarray): Current centroids. Shape (k, n_features) for 2D data,
                                    or (k, time_steps, n_features_ts) for 3D time series data.

        Returns:
            np.ndarray: Distances of shape (n_samples, k).
        """
        n_samples = x.shape[0]
        k_centroids = centroids.shape[0]
        distances = np.empty((n_samples, k_centroids), dtype=np.float64)

        if self.distance_metric == "euclidean":
            if x.ndim == 2:
                for i in range(k_centroids):
                    distances[:, i] = np.linalg.norm(x - centroids[i], axis=1)
            # Time series data
            elif x.ndim == 3:
                # Reshape for broadcasting: x (N,1,T,F), centroids (1,K,T,F)
                diff = x[:, np.newaxis, :, :] - centroids[np.newaxis, :, :, :]
                distances = np.linalg.norm(diff, axis=(2, 3))
            else:
                raise ValueError(
                    "Euclidean distance supports 2D or 3D input data.")

        elif self.distance_metric == "manhattan":
            if x.ndim == 2:
                for i in range(k_centroids):
                    distances[:, i] = np.sum(np.abs(x - centroids[i]), axis=1)

            elif x.ndim == 3:  # (n_samples, time_steps, n_features_ts)
                diff = x[:, np.newaxis, :, :] - centroids[np.newaxis, :, :, :]
                distances = np.sum(np.abs(diff), axis=(2, 3))

            else:  #
                raise ValueError(
                    "Manhattan distance supports 2D or 3D input data.")

        elif self.distance_metric == "dtw":
            distances = self._dtw(x, centroids)

        else:
            raise ValueError(
                f"Unsupported distance metric: {self.distance_metric}")

        return distances

    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.

        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)

        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """

        n_samples = x.shape[0]
        k_centroids = centroids.shape[0]
        dtw_distances = np.empty((n_samples, k_centroids), dtype=np.float64)

        # Prepare data: ensure 3D for consistent processing (N, T, F)
        # If input is 2D (N, T), treat as (N, T, 1)
        x_proc = x if x.ndim == 3 else x[:, :, np.newaxis]
        centroids_proc = centroids if centroids.ndim == 3 else centroids[:, :, np.newaxis]

        if x_proc.shape[2] != centroids_proc.shape[2]:  # Check feature consistency
            raise ValueError(
                "DTW: x and centroids must have the same number of features.")

        n_features_ts = x_proc.shape[2]

        for i in range(n_samples):
            for j in range(k_centroids):
                sample_ts_multivar = x_proc[i]      # (T, F_ts)
                centroid_ts_multivar = centroids_proc[j]  # (T, F_ts)
                current_total_dtw_dist = 0.0
                for feat_idx in range(n_features_ts):
                    s1 = sample_ts_multivar[:, feat_idx]
                    s2 = centroid_ts_multivar[:, feat_idx]
                    # dtaidistance.dtw.distance_fast is efficient
                    current_total_dtw_dist += dtw.distance_fast(
                        s1, s2, use_pruning=True)
                dtw_distances[i, j] = current_total_dtw_dist
        return dtw_distances

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.

        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).

        Returns:
            MyKMeans: Fitted estimator instance.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError(
                "Input data must be a numpy array or a pandas DataFrame")

        # Add dimension check after type conversion and before other operations
        if not (x.ndim == 2 or x.ndim == 3):
            raise ValueError("Input data must be a 2D or 3D array")
        if x.shape[0] < self.k:
            raise ValueError(
                f"Number of samples ({x.shape[0]}) must be at least k ({self.k}).")

        self.centroids = self._initialize_centroids(x)

        # Use tqdm for progress bar
        for iteration in tqdm(range(self.max_iter), desc="K-Means Iterations", leave=False):
            old_centroids = self.centroids.copy()

            # 1. Assignment step: Assign samples to the closest centroid
            distances = self._compute_distance(x, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)

            # 2. Update step: Recalculate centroids
            new_centroids = np.empty_like(self.centroids)
            for i in range(self.k):
                cluster_points = x[self.labels_ == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty cluster
                    logging.warning(
                        f"Cluster {i} became empty at iteration {iteration}.")

                    rng = np.random.default_rng(42)
                    new_centroids[i] = x[rng.choice(
                        x.shape[0])].copy()

            self.centroids = new_centroids

            if np.allclose(old_centroids, self.centroids):
                logging.info(f"Converged at iteration {iteration+1}.")
                break
        else:
            logging.info(
                f"K-Means did not converge within {self.max_iter} iterations.")

        final_distances_to_assigned_centroids = self._compute_distance(
            x, self.centroids)
        min_distances = np.min(final_distances_to_assigned_centroids, axis=1)

        if self.distance_metric in ["euclidean", "manhattan"]:
            self.inertia_ = np.sum(min_distances**2)
        elif self.distance_metric == "dtw":
            self.inertia_ = np.sum(min_distances)

        return self

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.

        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)

        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError(
                "Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)
