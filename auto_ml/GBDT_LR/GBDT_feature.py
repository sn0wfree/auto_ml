# coding=utf-8
import numpy as np
import random
# import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from xgboost.sklearn import XGBClassifier

np.random.seed(10)

X, Y = make_classification(n_samples=1000, n_features=30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=233, test_size=0.5)
X_train, X_train_lr, Y_train, Y_train_lr = train_test_split(X_train, Y_train, random_state=233, test_size=0.2)


def RandomForestLR():
    RF = RandomForestClassifier(n_estimators=100, max_depth=4)
    RF.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(RF.apply(X_train))
    LR = LogisticRegression()
    LR.fit(OHE.transform(RF.apply(X_train_lr)), Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(RF.apply(X_test)))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('RandomForest + LogisticRegression: ', auc)
    return fpr, tpr


# def XGBoostLR():
#     XGB = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
#     XGB.fit(X_train, Y_train)
#     OHE = OneHotEncoder()
#     OHE.fit(XGB.apply(X_train))
#     LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
#     LR.fit(OHE.transform(XGB.apply(X_train_lr)), Y_train_lr)
#     Y_pred = LR.predict_proba(OHE.transform(XGB.apply(X_test)))[:, 1]
#     fpr, tpr, _ = roc_curve(Y_test, Y_pred)
#     auc = roc_auc_score(Y_test, Y_pred)
#     print('XGBoost + LogisticRegression: ', auc)
#     return fpr, tpr


def GBDTLR():
    GBDT = GradientBoostingClassifier(n_estimators=10)
    GBDT.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(GBDT.apply(X_train)[:, :, 0])
    LR = LogisticRegression()
    LR.fit(OHE.transform(GBDT.apply(X_train_lr)[:, :, 0]), Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('GradientBoosting + LogisticRegression: ', auc)
    return fpr, tpr


def LR():
    LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('LogisticRegression: ', auc)
    return fpr, tpr


# def XGBoost():
#     XGB = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
#     XGB.fit(X_train, Y_train)
#     Y_pred = XGB.predict_proba(X_test)[:, 1]
#     fpr, tpr, _ = roc_curve(Y_test, Y_pred)
#     auc = roc_auc_score(Y_test, Y_pred)
#     print('XGBoost: ', auc)
#     return fpr, tpr


if __name__ == '__main__':
    RF = RandomForestClassifier(n_estimators=100, max_depth=4)
    RF.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(RF.apply(X_train))
    print(1)
