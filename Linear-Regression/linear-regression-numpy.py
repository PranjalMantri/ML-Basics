import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Weights and biases will be initalised to 0 in fit()
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Samples refer to the total data points, features refers to the number of input variables
        n_samples, n_features = X.shape

        # Number of weight depends on the number of features
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Implementing the equation for linear regression, y = wx + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Effecient way of getting the derivate of error for weights and biases (Basically Gradient Descent)
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    # Predicting involves fitting the new data point to the line, we have already computed 
    # the optimal values for each parameter in fit function
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    # Mean squared Error: It is basically the average of squares of difference between the predicited value and the actual value
    def mse(self, y, y_pred):
        mse = np.mean((y_pred - y)**2)
        return mse
    

if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 7, 11, 15, 19])

    model = LinearRegression()
    model.fit(X, y)

    x_test = np.array([[6]])
    y_pred = model.predict(x_test)
    
    print("Predictions: ", y_pred)
    print("True values: ", y)

    mse = model.mse(y, y_pred)
    print("Mean Squared Error: ", mse)