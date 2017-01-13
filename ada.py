import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

np.random.seed(0)

digits = load_digits()

x, y = digits.data, digits.target

depths = [1, 2, 10]

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d)
    adab = AdaBoostClassifier(tree)
    scores = np.mean(cross_val_score(adab, x, y, cv=6))
    print(d, scores)


n_boot = [i for i in range(5, 105, 5)]
learning_scores = np.zeros(len(n_boot))
test_scores = np.zeros(len(n_boot))
tree = DecisionTreeClassifier()
for i, n_b in enumerate(n_boot):
    adab = AdaBoostClassifier(tree, n_estimators=n_b)
    adab.fit(x, y)
    learning_scores[i] = adab.score(x, y)
    test_scores[i] = np.mean(cross_val_score(adab, x, y))

print(learning_scores)
print(test_scores)

plt.figure(1)
plt.plot(n_boot, learning_scores, c="red", label="train scores")
plt.plot(n_boot, test_scores, c="blue", label="test scores")
plt.legend()
plt.show()

    Contact GitHub API Training Shop Blog About 


