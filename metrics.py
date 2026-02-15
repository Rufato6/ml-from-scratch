"""
Evaluation metrics for machine learning algorithms.

This module provides basic performance metrics used for classification
and regression tasks.  Functions operate on NumPy arrays or lists and
are written to be self‑contained without external dependencies.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable


def accuracy_score(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute the classification accuracy.

    Accuracy is the ratio of correct predictions to the total number of
    predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


def precision_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute precision for binary classification.

    Precision is the ratio of true positives to the sum of true
    positives and false positives.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    if true_pos + false_pos == 0:
        return 0.0
    return float(true_pos / (true_pos + false_pos))


def recall_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute recall for binary classification.

    Recall is the ratio of true positives to the sum of true
    positives and false negatives.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))
    if true_pos + false_neg == 0:
        return 0.0
    return float(true_pos / (true_pos + false_neg))


def f1_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute the F1 score for binary classification.

    The F1 score is the harmonic mean of precision and recall.
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mean_squared_error(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute the mean squared error (MSE).

    MSE is the average of the squared differences between actual and
    predicted values.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute the mean absolute error (MAE)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute the coefficient of determination (R² score)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot