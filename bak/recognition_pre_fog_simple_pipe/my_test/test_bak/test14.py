import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC,SVR
clf = SVC(gamma='auto')
clf.fit(X, y)


print(clf.support_, clf.dual_coef_ ,clf.get_params())

print(clf.predict([[-0.8, -1]]))
import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)

print(clf.predict(X[2:3]))
clf.feature_log_prob_