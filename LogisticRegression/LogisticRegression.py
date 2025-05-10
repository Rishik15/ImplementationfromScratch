import numpy as np

class LogisticRegression():

    def __init__(self, lr = 0.4, iter = 3000):
        self.lr = lr
        self.iter = iter
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            linear = np.dot(X_train, self.weights) + self.bias
            predictions = self.sigmoid(linear)


            dw = (1/n_samples) * np.dot(X_train.T, (predictions - y_train))
            db = (1/n_samples) * np.sum(predictions - y_train)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X_test):
        linear = np.dot(X_test, self.weights) + self.bias
        predictions = self.sigmoid(linear)
        return np.where(predictions > 0.5, 1, 0)

    @staticmethod
    def sigmoid(linear):
        return np.exp(np.fmin(linear, 0)) / (1 + np.exp(-np.abs(linear)))
