# Logistic Regression

# Only difference from linear regression is the sigmoid function and calculating classifaction at the end

# Create probabilities instead of a specific value
# 	- Put our value inside sigmoid function
# 	- Get probability distribution between 0 and 1

# yhat = h(x) = 1/(1 + e^(-(wx + b)))
# sigmoid s(x) = 1/(1 + e^(-x))

# Instead of MSE, we use cross entropy
# J(w,b) = J(theta) = 1/N * sum(y^i * log(h(x^i)) + (1 - y^i) * log(1 - h(x^i)))

# Gradient Descent
# 	Too low learning rate: never reaches minima
# 	Too high learning rate: bounces around minima, can't find it

import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class LogisticRegresson:

	def __init__(self, lr = 0.001, n_iters = 1000):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iters):
			# Get predictions for this round
			linear_pred = np.dot(X, self.weights) + self.bias
			y_pred = sigmoid(linear_pred)

			# Calculate gradients
			dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
			db = (1/n_samples) * np.sum(y_pred - y)

			# Update weights and bias
			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db

	def predict(self, X):
		linear_pred = np.dot(X, self.weights) + self.bias
		y_pred = sigmoid(linear_pred)
		class_pred = [0 if y<=0.5 else 1 for y in y_pred]
		return class_pred