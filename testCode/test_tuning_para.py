import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']


gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

# param_test1 = {'n_estimators': list(range(20,81,10))}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
#                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


params={'learning_rate':[0.05,0.1,0.2], 'max_depth':[x for x in range(2,10,1)], 'min_samples_leaf':
                [x for x in range(5,30,5)], 'n_estimators':[x for x in range(20,101,10)]}
clf = GradientBoostingClassifier()
grid = GridSearchCV(clf, params, cv=10, scoring="roc_auc")
grid.fit(X, y)
print(grid.best_score_)   #查看最佳分数(此处为f1_score)
print(grid.best_params_)   #查看最佳参数)
print(grid.best_estimator_)



