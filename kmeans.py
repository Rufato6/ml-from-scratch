"""
K‑Means clustering algorithm implementation.

This class implements Lloyd's algorithm for clustering points into k
clusters.  Centroids are initialized by randomly sampling points from
the dataset.  At each iteration, points are assigned to the nearest
centroid and then centroids are recomputed as the mean of assigned
points.  The process stops when centroids converge or when the
maximum number of iterations is reached.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable


class KMeans:
    """K‑Means clustering.

    Parameters
    ----------
    k : int, default=3
        Number of clusters.

    max_iters : int, default=100
        Maximum number of iterations of the k‑means algorithm.

    tolerance : float, default=1e‑4
        Convergence tolerance.  If the change in centroids is below this
        threshold, the algorithm stops early.
    """

    def __init__(self, k: int = 3, max_iters: int = 100, tolerance: float = 1e-4) -> None:
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: Iterable[Iterable[float]]) -> None:
        """Compute k‑means clustering.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training instances to cluster.
        """
        X = np.array(X, dtype=float)
        n_samples, _ = X.shape
        # Randomly initialize centroids as k distinct samples
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # Assign each sample to the nearest centroid
            distances = self._euclidean_distance(X, self.centroids)
            clusters = np.argmin(distances, axis=1)
            # Compute new centroids as the mean of assigned samples
            new_centroids = np.zeros_like(self.centroids)
            for idx in range(self.k):
                points = X[clusters == idx]
                if len(points) > 0:
                    new_centroids[idx] = points.mean(axis=0)
                else:
                    # If no points assigned, keep previous centroid
                    new_centroids[idx] = self.centroids[idx]
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tolerance):
                break
            self.centroids = new_centroids

        self.labels_ = clusters

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            New samples to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X, dtype=float)
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    @staticmethod
    def _euclidean_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between each sample and each centroid."""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for idx, centroid in enumerate(centroids):
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)
        return distances