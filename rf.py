import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Parameters
# Load data
boston = load_boston()
diabetes = load_diabetes()
iris = load_iris()
digits = load_digits()

datasets = [boston, diabetes, iris, digits]
ds_names = ["boston", "diabetes", "iris", "digits"]

for i, ds in enumerate(datasets):
    x, y = ds.data, ds.target
    if i <= 1:
        rf = RandomForestRegressor()
        svc = SVR()
    else:
        rf = RandomForestClassifier()
        svc = SVC()
    score_rf = np.mean(cross_val_score(rf, x, y, cv=7))
    score_svc = np.mean(cross_val_score(svc, x, y, cv=7))
    print(ds_names[i], "random forest : ", score_rf, "svc : ", score_svc)

"""
n_estimators = 2
plot_colors = "bry"
plot_step = 0.02
x_unscaled, y = iris.data[:, :2], iris.target
x = preprocessing.scale(x_unscaled)
rf = RandomForestClassifier(n_estimators = n_estimators)
rf.fit(x, y)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
np.arange(y_min, y_max, plot_step))
# z = z.reshape(xx.shape)
z = np.zeros(xx.shape)
plt.figure()
for i, tree in enumerate(rf.estimators_):
# TODO use predict to obtain the probabilities you will store in Z
    z[i] = tree.predict_proba
    cs = plt.contourf(xx, yy, z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
    plt.axis("tight")
"""

x, y = iris.data[:, :2], iris.target
depths = [i for i in range(1, 31, 1)]
score_tree = np.zeros(len(depths))
cross_val_tree = np.zeros(len(depths))
score_rf = np.zeros(len(depths))
cross_val_rf = np.zeros(len(depths))
for i, d in enumerate(depths):
    tree = DecisionTreeClassifier(max_depth=d)
    rf = RandomForestClassifier(max_depth=d)
    tree.fit(x, y)
    score_tree[i] = tree.score(x, y)
    cross_val_tree[i] = np.mean(cross_val_score(tree, x, y, cv=6))
    rf.fit(x, y)
    score_rf[i] = rf.score(x, y)
    cross_val_rf[i] = np.mean(cross_val_score(rf, x, y, cv=6))

plt.figure(1)
plt.plot(depths, score_tree, c="red", label="score")
plt.plot(depths, cross_val_tree, c="blue", label="cross val score")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(depths, score_rf, c="red", label="score")
plt.plot(depths, cross_val_rf, c="blue", label="cross val score")
plt.legend()
plt.show()

"""
model = RandomForestClassifier(n_estimators=n_estimators)
clf = model.fit(X, y)
# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
np.arange(y_min, y_max, plot_step))
2plt.figure()
for tree in model.estimators_:
# TODO use predict to obtain the probabilities you will store in Z
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
plt.axis("tight")
# Plot the training points
for i, c in zip(range(3), plot_colors):
idx = np.where(y == i)
plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
cmap=plt.cm.Paired)
plt.legend(scatterpoints=1)
plt.show()
"""
