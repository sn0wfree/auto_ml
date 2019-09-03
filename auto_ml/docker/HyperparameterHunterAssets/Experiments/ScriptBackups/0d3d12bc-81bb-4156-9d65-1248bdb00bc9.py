# -*- coding:utf-8 -*-


from hyperparameter_hunter import Environment, CVExperiment
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

env = Environment(
    train_dataset=df,  # Add holdout/test dataframes, too
    results_path='.',  # Where your result files will go
    metrics=['roc_auc_score'],  # Callables, or strings referring to `sklearn.metrics`
    cv_type=StratifiedKFold,  # Class, or string in `sklearn.model_selection`
    cv_params=dict(n_splits=5, shuffle=True, random_state=32)
)
experiment = CVExperiment(
    model_initializer=LinearSVC,  # (Or any of the dozens of other SK-Learn algorithms)
    model_init_params=dict(penalty='l1', C=0.9,dual=False)  # Default values used and recorded for kwargs not given
)
if __name__ == '__main__':
    df = load_breast_cancer()
    print(1)
