"""
Gaussian Naïve Bayes classifier.

This implementation assumes that the features follow a normal (Gaussian)
distribution.  It computes the mean and variance of each feature for
each class during training and uses these statistics to calculate
likelihoods when predicting class probabilities.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Any


class GaussianNaiveBayes:
    """Gaussian Naïve Bayes classifier.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    priors_ : dict
        Prior probabilities of each class.

    mean_ : dict
        Mean of each feature per class.

    var_ : dict
        Variance of each feature per class.
    """

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]) -> None:
        """Fit the Gaussian Naïve Bayes classifier.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Class labels.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        # Initialize dictionaries to hold statistics
        self.priors_ = {}
        self.mean_ = {}
        self.var_ = {}
        n_samples, _ = X.shape
        for cls in self.classes_:
            X_c = X[y == cls]
            self.priors_[cls] = X_c.shape[0] / n_samples
            self.mean_[cls] = X_c.mean(axis=0)
            # Add small value to variance for numerical stability
            self.var_[cls] = X_c.var(axis=0) + 1e-9

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted class labels for samples in X.
        """
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> Any:
        posteriors = []
        for cls in self.classes_:
            prior = np.log(self.priors_[cls])
            # Compute log likelihood for each feature and sum them
            class_conditional = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[cls]))
            class_conditional -= 0.5 * np.sum(((x - self.mean_[cls]) ** 2) / self.var_[cls])
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes_[np.argmax(posteriors)]