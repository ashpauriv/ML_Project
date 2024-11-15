# main_script.py
import pandas as pd
import numpy as np
from models.linear_regression import LinearRegression
from models.gradient_boosting import GradientBoostingRegressor
from models.q_learning import run_q_learning  # Import Q-Learning simulation
from sklearn.metrics import mean_squared_error

# Load and preprocess data
X_train = pd.read_csv('X_train.csv').to_numpy()
X_test = pd.read_csv('X_test.csv').to_numpy()
y_train = pd.read_csv('y_train.csv').squeeze().to_numpy()
y_test = pd.read_csv('y_test.csv').squeeze().to_numpy()

# Handle NaNs
X_train = pd.DataFrame(X_train).fillna(X_train.mean()).to_numpy()
X_test = pd.DataFrame(X_test).fillna(X_test.mean()).to_numpy()
y_train = pd.Series(y_train).fillna(y_train.mean()).to_numpy()
y_test = pd.Series(y_test).fillna(y_test.mean()).to_numpy()

# Normalize
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Check for NaNs or inf
assert not np.isnan(X_train).any() and not np.isinf(X_train).any(), "NaNs or inf in X_train"
assert not np.isnan(X_test).any() and not np.isinf(X_test).any(), "NaNs or inf in X_test"
assert not np.isnan(y_train).any() and not np.isinf(y_train).any(), "NaNs or inf in y_train"
assert not np.isnan(y_test).any() and not np.isinf(y_test).any(), "NaNs or inf in y_test"

# Dictionary for results
results = {}

# Linear Regression
print("Running Linear Regression...")
lr_model = LinearRegression(learning_rate=0.01, iterations=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
assert not np.isnan(lr_predictions).any(), "NaNs in Linear Regression predictions"
results['Linear Regression'] = mean_squared_error(y_test, lr_predictions)
print(f"Linear Regression MSE: {results['Linear Regression']}")

# Gradient Boosting
print("Running Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
assert not np.isnan(gb_predictions).any(), "NaNs in Gradient Boosting predictions"
results['Gradient Boosting'] = mean_squared_error(y_test, gb_predictions)
print(f"Gradient Boosting MSE: {results['Gradient Boosting']}")

# Q-Learning
q_table = run_q_learning()  # Call the Q-Learning simulation and get the Q-table

# Display Results
print("\nModel Performances:")
for model, mse in results.items():
    print(f"{model}: MSE = {mse}")

print("\nQ-Learning Q-Table:")
print(q_table)
