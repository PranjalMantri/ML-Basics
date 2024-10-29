import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 7, 11, 15, 19])

model = LinearRegression()
model.fit(X, y)

x_test = np.array([[6]])
y_pred = model.predict(x_test)

print(f"Y_Pred: {y_pred}")