import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k = 3, metric = "euclidian"):
        self.k = k
        self.metric = metric
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x_test):
        if(self.metric == "euclidian"):
            distances = [self.euclidean_distance(x_test, x_train) for x_train in self.X_train]
        elif(self.metric == "manhattan"):
            distances = [self.manhattan_distance(x_test, x_train) for x_train in self.X_train]
        else:
            distances = [self.cosine_distance(x_test, x_train) for x_train in self.X_train]
        
        indices = np.argsort(distances)[:self.k]
        nearest_neighbors = [self.y_train[i] for i in indices]

        most_common = Counter(nearest_neighbors).most_common()

        return most_common[0][0]

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def cosine_distance(x1, x2):
        dot_product = np.dot(x1,x2)
        x1_mag = np.linalg.norm(x1)
        x2_mag = np.linalg.norm(x2)

        if x1_mag == 0 or x2_mag == 0:
            return 1

        return 1 - (dot_product/(x1_mag*x2_mag))
    
    @staticmethod
    def manhattan_distance(x1, x2):
        return np.linalg.norm(x1-x2, ord = 1)
    



    
