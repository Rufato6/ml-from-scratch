"""
Implementation of Linear Regression using gradient descent.

The LinearRegression class fits a linear model to data by minimizing the
mean squared error loss function.  It uses gradient descent to adjust
weights and bias over a specified number of iterations.  Because only
NumPy is used for calculations, this implementation is suitable for
educational purposes and small datasets.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Optional


class LinearRegression:
    """Ordinary least squares linear regression with gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size applied to the gradient on each update.

    n_iters : int, default=1000
        Number of passes over the training data during optimization.
    """

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        # Model parameters initialized during fitting
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> None:
        """Fit linear model to training data.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Target values.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        # Perform gradient descent optimization
        for _ in range(self.n_iters):
            # Compute predictions
            y_predicted = np.dot(X, self.weights) + self.bias
            # Compute gradients
            dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict target values for new samples.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.weights is None:
            raise RuntimeError("You must fit the model before making predictions.")
        X = np.array(X, dtype=float)
        return np.dot(X, self.weights) + self.bias