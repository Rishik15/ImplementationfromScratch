from node import Node
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X_train, y_train):
        self.n_features = X_train.shape[1] if not self.n_features else min(X_train.shape[1], self.n_features)
        self.root = self.growTree(X_train, y_train)

    def growTree(self, X_train, y_train, depth = 0):
        n_samples, n_featrs = X_train.shape
        n_labels = len(np.unique(y_train))

        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y_train)
            return Node(value=leaf_value)
        
        feat_ids = np.random.choice(n_featrs, self.n_features, replace = False)

        best_thresh, best_feature = self.bestSplit(X_train, y_train, feat_ids)

        left_ids, right_ids = self.split(X_train[:, best_feature], best_thresh)
        left = self.growTree(X_train[left_ids, :], y_train[left_ids], depth+1)
        right = self.growTree(X_train[right_ids, :], y_train[right_ids], depth+1)

        return Node(best_feature, best_thresh, left, right)

    def bestSplit(self, X_train, y_train, feat_ids):
        best_gain = -1
        split_thr, split_idx = None, None

        for feat_id in feat_ids:
            X_column = X_train[:, feat_id]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self.calculate_InformationGain(X_column, y_train, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_thr = threshold
        
        return split_thr, split_idx
    
    def calculate_InformationGain(self, X_column, y_train, threshold):
        parent_entropy = self.entropy(y_train)

        left_child_ids, right_child_ids = self.split(X_column, threshold)

        if len(left_child_ids) == 0 or len(right_child_ids) == 0:
            return 0
        
        n = len(y_train)
        n_l, n_r = len(left_child_ids), len(right_child_ids)

        e_l, e_r = self.entropy(y_train[left_child_ids]), self.entropy(y_train[right_child_ids])

        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy

        return information_gain

    
    def split(self, X_column, threshold):
        left_child_ids = np.argwhere(X_column <= threshold).flatten()
        right_child_ids = np.argwhere(X_column > threshold).flatten()

        return left_child_ids, right_child_ids


    def entropy(self, y_train):
        hist = np.bincount(y_train)
        ps = hist / len(y_train)

        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def most_common_label(self, y_train):
        counter = Counter(y_train)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X_test):
        predictions = np.array([self.traverse_tree(x, self.root) for x in X_test])
        return predictions

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        
        return self.traverse_tree(x, node.right)
        
