"""
Utilities for splitting data sets into train and test partitions.

This module provides a simple implementation of the popular
``train_test_split`` function.  It shuffles the data (optionally with a
reproducible random seed) before splitting so that both the training
and testing sets are representative of the overall distribution.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Any


def train_test_split(
    X: Iterable[Iterable[Any]],
    y: Iterable[Any],
    test_size: float = 0.2,
    random_state: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array‑like of shape (n_samples, n_features)
        Input features.

    y : array‑like of shape (n_samples,)
        Target labels.

    test_size : float, default=0.2
        Fraction of the dataset to include in the test split.  Must be
        between 0.0 and 1.0.

    random_state : int or None, default=None
        Controls the shuffling applied to the data before splitting.  Pass
        an int for reproducible output across multiple function calls.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Partitioned training and testing data.
    """
    X = np.array(X)
    y = np.array(y)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    n_samples = len(X)
    # Shuffle data indices
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Determine split index
    test_count = int(n_samples * test_size)
    # Partition
    X_test = X[:test_count]
    y_test = y[:test_count]
    X_train = X[test_count:]
    y_train = y[test_count:]
    return X_train, X_test, y_train, y_test