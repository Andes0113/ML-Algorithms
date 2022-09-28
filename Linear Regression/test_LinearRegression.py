import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def MeanSquaredError(y_test, y_pred):
	return np.mean((y_test - y_pred)**2)

LinReg = LinearRegression(lr=0.008)

LinReg.fit(X_train, y_train)
y_pred = LinReg.predict(X_train)

mse = MeanSquaredError(y_train, y_pred)
print(mse)

y_pred_line = LinReg.predict(X)
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, s=10)
m2 = plt.scatter(X_test, y_test, s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()