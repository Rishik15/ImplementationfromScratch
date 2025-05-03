import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN

digits = datasets.load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNN(k=3, metric="cosine")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)

print("Accuracy = ", accuracy)

