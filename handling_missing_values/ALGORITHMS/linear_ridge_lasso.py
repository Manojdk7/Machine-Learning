# Linear Regression → Ridge Regression → Lasso Regression

# ==========================================
# 1. Import California Housing Dataset
# ==========================================

from sklearn.datasets import fetch_california_housing

df = fetch_california_housing()


# ==========================================
# 2. Convert Dataset into Pandas DataFrame
# ==========================================

import pandas as pd

dataset = pd.DataFrame(df.data)

dataset.columns = df.feature_names

print(dataset.head())


# ==========================================
# 3. Add Target Variable (PRICE)
# ==========================================

dataset["PRICE"] = df.target

print(dataset.head())


# ==========================================
# 4. Divide Dataset into X and y
# ==========================================

# Independent features
X = dataset.iloc[:, :-1]

# Dependent feature / Target
y = dataset.iloc[:, -1]


# ==========================================
# 5. Linear Regression
# ==========================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

le = LinearRegression()

mse = cross_val_score(
    le,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=5
)

print("Linear Regression MSE:", mse)

mean_mse = np.mean(mse)

print("Linear Regression Mean MSE:", mean_mse)


# ==========================================
# 6. Ridge Regression
# ==========================================

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

re = Ridge()

params = {
    "alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]
}

ridge_regression = GridSearchCV(
    re,
    params,
    scoring="neg_mean_squared_error",
    cv=5
)

ridge_regression.fit(X, y)


print("\nRidge Regression")
print("Best Alpha:", ridge_regression.best_params_)
print("Best Score:", ridge_regression.best_score_)


# ==========================================
# 7. Lasso Regression
# ==========================================

from sklearn.linear_model import Lasso

lasso = Lasso(max_iter=10000)

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 20]
}

lasso_regression = GridSearchCV(
    lasso,
    params,
    scoring="neg_mean_squared_error",
    cv=5
)

lasso_regression.fit(X, y)


print("\nLasso Regression")
print("Best Alpha:", lasso_regression.best_params_)
print("Best Score:", lasso_regression.best_score_)