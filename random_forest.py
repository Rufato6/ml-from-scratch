"""
Random Forest classifier.

The RandomForestClassifier is an ensemble method that constructs
multiple decision trees on bootstrapped samples of the data and
aggregates their predictions.  Optionally, each tree can use a random
subset of the features to further decorrelate trees.  The final
prediction is obtained via majority vote across individual trees.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Any
from collections import Counter

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """An ensemble classifier of multiple decision trees.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of each individual tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    max_features : int or None, default=None
        The number of features to randomly select when looking for the best split.
        If ``None``, all features are used.  A typical default is ``sqrt(n_features)``.
    """

    def __init__(self, n_estimators: int = 10, max_depth: int | None = None,
                 min_samples_split: int = 2, max_features: int | None = None) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees: List[DecisionTreeClassifier] = []
        self.feature_indices: List[np.ndarray] = []

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]) -> None:
        """Fit the random forest classifier from the training set.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Target values.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples, n_features = X.shape
        # Determine number of features to sample for each tree
        if self.max_features is None:
            # default to sqrt of total features
            features_per_tree = int(np.sqrt(n_features))
        else:
            features_per_tree = min(self.max_features, n_features)

        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample of data
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            # Random subset of features
            feature_idx = np.random.choice(n_features, features_per_tree, replace=False)
            X_sample = X[sample_indices][:, feature_idx]
            y_sample = y[sample_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict class for X.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.array(X, dtype=float)
        # Collect predictions from each tree
        tree_preds = []
        for tree, feat_idx in zip(self.trees, self.feature_indices):
            pred = tree.predict(X[:, feat_idx])
            tree_preds.append(pred)
        # Transpose to get predictions per sample
        tree_preds = np.array(tree_preds).T
        # Majority vote
        majority_votes = []
        for preds in tree_preds:
            vote = Counter(preds).most_common(1)[0][0]
            majority_votes.append(vote)
        return np.array(majority_votes)