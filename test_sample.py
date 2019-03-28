# coding=utf8
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np
from auto_ml_core import Models, test_dataset

regressors = {'svr': 1,
              'knn': 2,
              'random_forest': 3,
              'extra_trees': 4,
              'ada_boost': 5,
              'gradient_boosting': 6,
              'sgd': 7,
              'xgboost': 9}

classifiers = {'svc': [6],
               'knn': [5, 7],
               'random_forest': 3,
               'extra_trees': 2,
               'ada_boost': [1, 2, 8],
               'gradient_boosting': [4],
               'sgd': [3, 9],
               'xgboost': 9}


# Download the data and split into training and test sets
def test():
    iris = load_iris()

    X = iris.data
    y = iris.target

    test_size = int(0.2 * len(y))
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-test_size]]
    y_train = y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_test = y[indices[-test_size:]]

    # Instantiate a HyperoptEstimator with the search space and number of evaluations

    estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                              preprocessing=any_preprocessing('my_pre'),
                              algo=tpe.suggest,
                              max_evals=100,
                              trial_timeout=120)

    # Search the hyperparameter space based on the data

    estim.fit(X_train, y_train)

    # Show the results

    print(estim.score(X_test, y_test))
    # 1.0

    print(estim.best_model())
    # {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
    #           max_depth=3, max_features='log2', max_leaf_nodes=None,
    #           min_impurity_decrease=0.0, min_impurity_split=None,
    #           min_samples_leaf=1, min_samples_split=2,
    #           min_weight_xfraction_leaf=0.0, n_estimators=13, n_jobs=1,
    #           oob_score=False, random_state=1, verbose=False,
    #           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}


if __name__ == '__main__':
    from hyperopt import fmin, tpe, hp

    best = fmin(fn=lambda x: x ** 2,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100)
    print(best)

    pass
