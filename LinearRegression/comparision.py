import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y= True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = np.mean((y_test - predictions)**2)

print(mse)

