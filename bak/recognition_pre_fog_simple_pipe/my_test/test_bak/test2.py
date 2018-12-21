"""
@Name:        test1
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/5
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)

X_new = selector.transform(X)
print(selector.support_,selector.ranking_,X_new.shape)