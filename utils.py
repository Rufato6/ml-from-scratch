"""
Utility functions for data preprocessing.

This module contains helper functions that are often useful when
preparing data for machine learning algorithms.  Currently, it
includes functions to normalize or standardize numerical features and
to create one‑hot encodings of categorical variables.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def normalize(X: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform min‑max scaling on each feature.

    Parameters
    ----------
    X : array‑like of shape (n_samples, n_features)
        Data to be normalized.

    Returns
    -------
    X_norm : ndarray
        Normalized data.
    min_vals : ndarray
        Per‑feature minimum values used for scaling.
    max_vals : ndarray
        Per‑feature maximum values used for scaling.
    """
    X = np.array(X, dtype=float)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    # Avoid division by zero
    scale = max_vals - min_vals
    scale[scale == 0] = 1.0
    X_norm = (X - min_vals) / scale
    return X_norm, min_vals, max_vals


def standardize(X: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    X : array‑like of shape (n_samples, n_features)
        Data to be standardized.

    Returns
    -------
    X_std : ndarray
        Standardized data.
    mean_vals : ndarray
        Per‑feature mean values.
    std_vals : ndarray
        Per‑feature standard deviation values.
    """
    X = np.array(X, dtype=float)
    mean_vals = X.mean(axis=0)
    std_vals = X.std(axis=0)
    # Avoid division by zero
    std_vals[std_vals == 0] = 1.0
    X_std = (X - mean_vals) / std_vals
    return X_std, mean_vals, std_vals


def one_hot_encode(y: Iterable) -> np.ndarray:
    """One‑hot encode a 1‑dimensional array of categorical variables.

    Parameters
    ----------
    y : array‑like of shape (n_samples,)
        Categorical target values.

    Returns
    -------
    one_hot : ndarray of shape (n_samples, n_classes)
        One‑hot encoded matrix.
    """
    y = np.array(y)
    classes = np.unique(y)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    one_hot = np.zeros((len(y), len(classes)), dtype=int)
    for i, label in enumerate(y):
        one_hot[i, class_to_index[label]] = 1
    return one_hot