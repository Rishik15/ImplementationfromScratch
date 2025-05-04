import numpy as np

class LinearRegressor():
    def __init__(self, lr = 0.01, iter = 1000):
        self.lr = lr
        self.iter = iter
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_pred = np.dot(X_train, self.weights) + self.bias

            dw = (1/n_samples) * 2 * np.dot(X_train.T, (y_pred - y_train))
            db = (1/n_samples) * 2 * np.sum(y_pred - y_train)

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict(self, X_test):
        return  np.dot(X_test, self.weights) + self.bias
