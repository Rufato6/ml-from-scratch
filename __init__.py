"""
Machine Learning From Scratch package
===================================

This package provides implementations of fundamental machine learning
algorithms written from scratch with NumPy.  The goal is to offer
transparent implementations that help learners understand how the
algorithms work internally.  You can import classes and utility
functions directly from the top‑level package for convenience.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .kmeans import KMeans
from .decision_tree import DecisionTreeClassifier
from .random_forest import RandomForestClassifier
from .naive_bayes import GaussianNaiveBayes
from .k_nearest_neighbors import KNearestNeighbors
from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from .model_selection import train_test_split
from .utils import normalize, standardize, one_hot_encode

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KMeans",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GaussianNaiveBayes",
    "KNearestNeighbors",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "train_test_split",
    "normalize",
    "standardize",
    "one_hot_encode",
]
