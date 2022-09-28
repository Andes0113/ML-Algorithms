from ctypes.wintypes import tagRECT
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from KNN import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = dataset.load_iris()
X, y = iris.data, iris.tagRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1344)

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

accuracy = np.sum(predictions == y_test)  / len(y_test)
print("Accuracy of KNN: ", accuracy)