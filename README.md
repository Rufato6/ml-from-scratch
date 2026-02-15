# Machine Learning From Scratch

Machine Learning From Scratch is an educational Python library that provides implementations of classical machine‑learning algorithms without relying on heavy external frameworks.  The goal of the project is to demonstrate how common algorithms work under the hood by building them using only NumPy and plain Python.  Each algorithm is implemented as a simple, well‑documented class with a familiar API (`fit`, `predict`, etc.) so you can quickly experiment and learn.

## Features

The library currently includes implementations of:

* **Linear Regression** – fits a linear model using gradient descent.
* **Logistic Regression** – binary classification using the logistic function.
* **K‑Means Clustering** – unsupervised clustering with configurable number of clusters.
* **Decision Tree Classifier** – builds a tree using the Gini impurity criterion.
* **Random Forest Classifier** – an ensemble of decision trees with bagging and random feature selection.
* **Gaussian Naïve Bayes** – probabilistic classifier assuming normal feature distributions.
* **K‑Nearest Neighbors** – a simple instance‑based classifier/regressor.
* **Evaluation Metrics** – accuracy, precision, recall, F1, mean squared error, etc.
* **Model Selection Utilities** – a `train_test_split` function for splitting data.
* **Utilities** – functions for normalizing and standardizing numeric features.

All algorithms are implemented in the directory `ml_from_scratch` and expose a consistent, scikit‑learn‑like interface.  The code is deliberately kept readable and commented so that students can follow the implementation details.

## Getting Started

1. **Clone the repository** or download the source.
2. Ensure that you have **Python 3.7 or later** installed along with **NumPy**.  The library depends only on NumPy for numerical operations:

   ```bash
   pip install numpy
   ```

3. Import the algorithms you need and train them on your data:

   ```python
   import numpy as np
   from ml_from_scratch.linear_regression import LinearRegression
   from ml_from_scratch.model_selection import train_test_split
   from ml_from_scratch.metrics import mean_squared_error, r2_score

   # generate synthetic data
   X = 2 * np.random.rand(100, 1)
   y = 4 + 3 * X[:, 0] + np.random.randn(100)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LinearRegression(learning_rate=0.01, n_iters=1000)
   model.fit(X_train, y_train)

   predictions = model.predict(X_test)
   print("MSE:", mean_squared_error(y_test, predictions))
   print("R²:", r2_score(y_test, predictions))
   ```

For classification algorithms, you can similarly use `accuracy_score`, `precision_score`, `recall_score` and `f1_score` from `ml_from_scratch.metrics`.

## Contributing

This project is aimed at learners who want to understand how machine‑learning algorithms work internally.  Feel free to open issues, suggest improvements or contribute additional algorithms!  Pull requests with explanations and tests are especially welcome.

## License

This project is licensed under the MIT License.  See the `LICENSE` file for details.