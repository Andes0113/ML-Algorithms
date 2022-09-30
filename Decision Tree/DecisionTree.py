# Decision Trees

# Decisions:
# 	How do we decide which feature to split on?
# 	Which point should I split on?
# 	When do we stop splitting?

# Steps:
# Given whole dataset:
# 	Calculate information gain with each possible split
# 	Divide set with that feature and value that gives most information gain
# 	Divide tree and do the same for all created branches until a stopping criteria is reached
# Testing:
# 	Follow the tree until you reach a leaf node
# 	Return the most common class label 

# Terms
# Information Gain: IG = Entropy(parent) - [weighted average] * Entropy(children)
# 	Basically, >IG = >Increase in order
# Entropy: E = -Sum(p(X) * log2(p(X)))
# 	p(X) = #x/n
# Stopping Criteria: maximum depth, minimum # of samples, minimum impurity decrease
# 	Maximum Depth: If certain amount of splits have happened, don't split
# 	Minimum # of Samples: If node has < certain # of samples, don't split
# 	Minimum Impurity Decrease: Min change of entropy for a split to happen

import numpy as np
from collections import Counter


class Node:
	# Included in a node: feature, threshold, left child, right child, value
	def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): 
		# *  makes it so that we have to pass value by name
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def isLeafNode(self):
		return self.value is not None


class DecisionTree:
	def __init__(self, min_samples=2, max_depth=100, n_features=None):
		self.min_samples = min_samples
		self.max_depth = max_depth
		self.n_features = n_features
		self.root=None


	# Training Function
	def fit(self, X, y):
		# Check n_features not > than # actual features
		self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
		self.root = self._grow_tree(X, y)

	# Generates tree
	def _grow_tree(self, X, y, depth=0):
		n_samples, n_feats = X.shape
		n_labels = len(np.unique(y))

		# Check the stopping criteria
		if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples):
			return Node(value=self._most_common_label(y))

		feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

		# Find best split
		best_feature, best_threshold  = self._best_split(X, y, feat_idxs)

		# Create child nodes
		left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
		left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
		right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
		return Node(best_feature, best_threshold, left, right)

	# Find best threshold to split at
	def _best_split(self, X, y, feat_idxs):
		best_gain = -1
		split_idx, split_threshold = None, None
		
		for feat_idx in feat_idxs:
			X_column = X[:, feat_idx]
			thresholds = np.unique(X_column)

			for threshold in thresholds:
				# Calculate information gain
				gain = self._information_gain(y, X_column, threshold)

				if gain > best_gain:
					best_gain = gain
					split_idx = feat_idx
					split_threshold= threshold
		return split_idx, split_threshold

	# Calculate information gain
	def _information_gain(self, y, X_column, threshold):
		# Parent entropy
		parent_entropy = self._entropy(y)

		# Create children
		left_idxs, right_idxs  = self._split(X_column, threshold)

		if len(left_idxs) == 0 or len(right_idxs) == 0:
			return 0

		# Calculate weighted avg. entropy of children
		n = len(y)
		n_left, n_right = len(left_idxs), len(right_idxs)
		e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
		child_entropy = (n_left / n) * e_left + (n_right/n) * e_right

		# Calculate information gain
		information_gain = parent_entropy - child_entropy
		return information_gain

	# Calculate left and right indices
	def _split(self, X_column, split_threshold):
		left_idxs = np.argwhere(X_column <= split_threshold).flatten() # All entries <= split threshold
		right_idxs = np.argwhere(X_column > split_threshold).flatten() # All entries > split threshold
		return left_idxs, right_idxs

	# Calculate entropy
	def _entropy(self, y):
		# E = -sum(p(x) * log2(p(x)))
		# p(x) = #x/n
		hist = np.bincount(y) # Get # occurences of x for each x in y
		pX = hist / len(y) # get p(x) for all x simultaneously
		return -np.sum([p * np.log(p) for p in pX if p > 0])

	# Get most common value for a node's group
	def _most_common_label(self, y):
		counter = Counter(y)
		return counter.most_common(1)[0][0]


	def predict(self, X):
		return np.array([self._traverse_tree(x, self.root) for x in X])

	# Go through tree to find correct label for x based on its features
	def _traverse_tree(self, x, node):
		if node.isLeafNode():
			return node.value
		
		if x[node.feature] <= node.threshold:
			return self._traverse_tree(x, node.left)
		else:
			return self._traverse_tree(x, node.right)