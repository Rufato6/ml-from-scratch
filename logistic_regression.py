"""
Logistic regression classifier implemented using gradient descent.

This class supports binary classification problems.  It optimizes the
logistic loss function via gradient descent and uses the sigmoid
function to map linear model outputs to probabilities.  The API is
similar to that of scikit‑learn for educational familiarity.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Optional


class LogisticRegression:
    """Binary logistic regression classifier.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size applied to the gradient on each update.

    n_iters : int, default=1000
        Number of passes over the training data during optimization.
    """

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Apply the logistic sigmoid function element‑wise."""
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> None:
        """Fit the logistic regression model.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Binary target values (0 or 1).
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            # Gradients
            dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / n_samples) * np.sum(y_predicted - y)
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Return probability estimates for the positive class.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        probabilities : ndarray of shape (n_samples,)
            Probability of each sample being in the positive class.
        """
        if self.weights is None:
            raise RuntimeError("The model has not been fitted yet.")
        X = np.array(X, dtype=float)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict binary class labels for samples in X.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return np.array([1 if p >= 0.5 else 0 for p in probabilities])