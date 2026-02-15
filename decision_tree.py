"""
Decision Tree classifier implementation using the Gini impurity criterion.

This implementation builds a binary decision tree that recursively
selects the best feature and threshold to split the data in order to
minimize the impurity of the resulting subsets.  The tree supports
classification tasks with discrete class labels.  Hyperparameters
allow the user to control the maximum depth and the minimum number of
samples required to perform a split.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Optional, Any


class _Node:
    """Internal class representing a node in the decision tree."""

    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['_Node'] = None, right: Optional['_Node'] = None,
                 *, value: Optional[Any] = None) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class prediction at leaf nodes


class DecisionTreeClassifier:
    """A simple binary decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree.  If ``None``, the tree will expand
        until all leaves are pure or contain fewer than ``min_samples_split``
        samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.root: Optional[_Node] = None

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]) -> None:
        """Build the decision tree classifier from the training set.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training input samples.

        y : array‑like of shape (n_samples,)
            Class labels.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.root = self._build_tree(X, y)

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
        if self.root is None:
            raise RuntimeError("The tree has not been fitted yet.")
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> _Node:
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or num_classes == 1:
            leaf_value = self._most_common_label(y)
            return _Node(value=leaf_value)

        # find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return _Node(value=leaf_value)

        # create child splits
        indices_left = X[:, best_feature] < best_threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        # recursively build subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return _Node(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_split:
            return None, None
        # compute impurity of current node
        best_gini = float('inf')
        best_idx, best_threshold = None, None
        # iterate over all features and possible split thresholds
        for feature_idx in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_idx], y)))
            thresholds = np.array(thresholds, dtype=float)
            classes = np.array(classes)
            # Precompute counts of class labels to left and right
            num_left = {}
            num_right = {c: np.sum(classes == c) for c in np.unique(classes)}
            left_count = 0
            for i in range(1, num_samples):
                label = classes[i - 1]
                num_left[label] = num_left.get(label, 0) + 1
                num_right[label] -= 1
                left_count += 1
                right_count = num_samples - left_count
                if thresholds[i] == thresholds[i - 1]:
                    continue
                gini_left = 1.0 - sum((num_left.get(c, 0) / left_count) ** 2 for c in num_left)
                gini_right = 1.0 - sum((num_right.get(c, 0) / right_count) ** 2 for c in num_right if num_right.get(c, 0) > 0)
                # weighted average of the impurity
                gini = (left_count / num_samples) * gini_left + (right_count / num_samples) * gini_right
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2.0
        return best_idx, best_threshold

    def _most_common_label(self, y: np.ndarray) -> Any:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _traverse_tree(self, x: np.ndarray, node: _Node) -> Any:
        # traverse recursively until a leaf is reached
        if node.value is not None:
            return node.value
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)