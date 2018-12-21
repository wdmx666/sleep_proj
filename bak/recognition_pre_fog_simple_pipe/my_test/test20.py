from __future__ import print_function
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target


pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
# Use combined features to transform dataset:
pipe.fit(X, y)
X_features = pipe.predict(X)
print("Combined space has", X_features.shape[0], "features")

