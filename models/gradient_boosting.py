# models/gradient_boosting.py
import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_value = y[left_mask].mean() if left_mask.any() else 0
                right_value = y[right_mask].mean() if right_mask.any() else 0
                error = np.sum((y[left_mask] - left_value) ** 2) + np.sum((y[right_mask] - right_value) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        return np.where(X[:, self.feature_index] <= self.threshold, self.left_value, self.right_value)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        residual = y.copy()
        for _ in range(self.n_estimators):
            model = DecisionStump()
            model.fit(X, residual)
            predictions = model.predict(X)
            residual -= self.learning_rate * predictions
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return predictions
