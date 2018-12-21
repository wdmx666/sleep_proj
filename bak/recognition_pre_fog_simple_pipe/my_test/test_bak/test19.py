from __future__ import print_function
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()
X, y = iris.data, iris.target
# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)
# Maybe some original features where good, too?
selection = SelectKBest(k=1)
# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
ct_f = ColumnTransformer([("pca", pca, [0,1,2,3]), ("univ_select", selection, [0,1,2,3])])
# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")

X_features2 = ct_f.fit(X, y).transform(X)
print("Combined space has", X_features2.shape[1], "features")

for i in range(20):
    print(X_features[i],"--> ",X_features2[i],X_features[i]-X_features2[i])
# svm = SVC(kernel="linear")
#
# # Do grid search over k, n_components and C:
#
# pipeline = Pipeline([("features", combined_features), ("svm", svm)])
#
# param_grid = dict(features__pca__n_components=[1, 2, 3],
#                   features__univ_select__k=[1, 2],
#                   svm__C=[0.1, 1, 10])
#
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10)
# grid_search.fit(X, y)
# print(grid_search.best_estimator_)