"""
K‑Nearest Neighbors (KNN) classifier.

This class implements the simplest form of instance‑based learning.  When
predicting the class label of a sample, it computes the distance to all
training samples and selects the majority class of the k nearest
neighbors.  Both classification and regression can be supported by
changing the voting mechanism; this implementation is for classification.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Any
from collections import Counter


class KNearestNeighbors:
    """K‑Nearest Neighbors classifier.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider when making predictions.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]) -> None:
        """Store the training data.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Target values.
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model has not been fitted yet.")
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            # Compute distances to all training samples
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get k nearest samples
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)