import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

# Create a random dataset
rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 1 * (0.5 - rng.rand(16))

n_estimators = 100  # L in the text
tree_max_depth =100
bagging_max_depth = 10

# TODO define the regressor by bagging stumes
tree = DecisionTreeRegressor(max_depth=tree_max_depth)
tree.fit(x, y)
bagging = BaggingRegressor(tree, n_estimators=n_estimators, bootstrap=False)
bagging.fit(x, y)

# Predict
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_tree = tree.predict(x_test)
y_bagging = bagging.predict(x_test)

plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', c="red", label="data")
plt.plot(x_test, y_tree, 'o', c="blue", label="tree")
plt.plot(x_test, y_bagging, 'o', c="green", label="bagging")
# TODO add plots for Bagging/Tree
plt.title("Decision Tree Regression")
plt.legend(loc=1, numpoints=1)
plt.show()

