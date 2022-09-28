# Linear Regression

# Understand the pattern (slope) of a given data set
# Assume: Linear pattern to data

# Mean Squared Error = 1 / N * sum((yi - (w*xi + b))^2)
#	w = weights, b = bias
# Find gradient of MSE to find minimum MSE

# Use gradient descent
# Find gradient using derivate, then multiply by learning rate and subtract it from the bias
# 	w = w - a * dw
# 	b = b - a * db

# Training: Init weights/bias at zero
# Given data point:
# 	Predict result using y = wx + b
# 	Calculate error
# 	Use gradient descent to figure out new weights and bias values
# 	Repeat n times
# Testing: 
# 	Put in values from the data point into y = wx + b

# Simplified derivatives: 
# 	dJ/dw = dw = 1/N * sum ( 2xi * (yhat - yi))
#	dJ/db = db = 1/N * sum ( 2 * (yhat - yi))

# Predict for all samples at once
# ypred = wX + b
# XT = [...] * ypred ([wx1 + b wx2 + b ... wxn + b]) = dw

import numpy as np

class LinearRegression:

	def __init__(self, lr = 0.001, n_iters = 1000):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	# Training
	def fit(self, X, y):
		#Initialization
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		# Training (Repeat n times)
		for _ in range(self.n_iters):
			# Get predictions for this round
			y_pred = np.dot(X, self.weights) + self.bias

			# Calculate gradient using difference between predictions and actual
			dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
			db = (1/n_samples) * np.sum(y_pred - y)

			# Update weights and bias
			# 	w = w - a * dw
			# 	b = b - a * db
			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db

	# Prediction
	def predict(self, X):
		# Predict using y = wX + b
		y_pred = np.dot(X, self.weights) + self.bias
		return y_pred