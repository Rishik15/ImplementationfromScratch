import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

data = datasets.load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = np.sum(predictions == y_test)/ len(y_test)

print(acc)