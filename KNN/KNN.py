# K Nearest Neighbors


# Given a data point:
# 	1. Calculate its distance from all other points in the dataset
# 	2. Get the closest K points
# 	3. Regression:
#		- Get the avg of their values
# 	   Classification:
#		- Get the label with majority vote
import numpy as np
from collections import Counter

# Get distances between points using numpy
def euclidean_distance(x1, x2):
	distance = np.sqrt(np.sum((x1-x2)**2))
	return distance

class KNN:
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y
	"""
	Calculate distance between this data point and all other data points
	Finding all closest, returning the label based on the K nearest neighbors
	"""
	def predict(self, X):
		predictions = [self._predict(x) for x in X]
		return predictions
	def _predict(self, x):
		# Compute distance for each point in set
		distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

		# Get closest k neighbors
		indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in indices]

		# Majority vote
		most_common = Counter(k_nearest_labels).most_common()
		return most_common[0][0]