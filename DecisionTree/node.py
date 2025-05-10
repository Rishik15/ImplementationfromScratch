class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None): # '*' makes it so you have to define Node object with 'value =' specifically
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None