from sklearn import svm, datasets,linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import GridSearchCV
import pandas
iris = datasets.load_iris()
groups = np.random.randint(0,10,150)
parameters = {'svc__kernel': ('linear', 'rbf'), 'svc__C':[1, 10]}

svc = svm.SVC(gamma="scale")
pipe = Pipeline([("a", StandardScaler()), ('b', StandardScaler()),('c', None)])

#clf = GridSearchCV(pipe, parameters, cv=5, n_jobs=4)
#clf.fit(iris.data, iris.target, groups=groups)
#clf.predict(iris.data)
pipe.fit(iris.data, iris.target)
res= pipe.transform(iris.data)
print(res.__class__)
#pipe.fit(iris.data, iris.target)
#pipe.predict(iris.data)
print(pipe.get_params())
#lg = linear_model.LogisticRegression(penalty='l2',max_iter=100)
#lg.fit(iris.data,iris.target)
sc =StandardScaler()
sc.fit_transform(iris.data)
print("-------------------------")
sc.fit_transform(iris.data)