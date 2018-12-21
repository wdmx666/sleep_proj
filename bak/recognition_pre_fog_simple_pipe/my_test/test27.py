# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
#
#
# logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,max_iter=10000, tol=1e-5, random_state=0)
# pca = PCA()
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
#
# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target
#
# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 20, 30, 40, 50, 64],
#     'logistic__alpha': np.logspace(-4, 4, 5),
# }
# #search = GridSearchCV(pipe, param_grid, iid=False, cv=5,return_train_score=False)
# #search.fit(X_digits, y_digits)
# #print("Best parameter (CV score=%0.3f):" % search.best_score_)
# #print(search.best_params_)
#
# # Plot the PCA spectrum
# #pca.fit(X_digits)
# pipe.fit(X_digits, y_digits)
#print(pipe)
import joblib
parrel = joblib.Parallel(n_jobs="ab")
parrel(joblib.delayed(sum)(i**2) for i in range(19))