# coding=utf8

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing, any_regressor

from hyperopt import tpe
import numpy as np


class DataSetParser(object):

    @staticmethod
    def datadict_parser(dataset_dict):
        X_train = dataset_dict['X_train']
        X_test = dataset_dict['X_test']
        y_train = dataset_dict['y_train']
        y_test = dataset_dict['y_test']
        return X_train, X_test, y_train, y_test

    @staticmethod
    def iris_test_dataset():
        from sklearn.datasets import load_iris
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

        dataset_dict = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return dataset_dict


class ModelBuilder(object):

    @classmethod
    def create_estimator(cls, params):
        """



        :param params:
        :return:
        """
        if 'regressor' in params.keys():
            if 'classifier' in params.keys():
                raise ValueError('params obtain two similar parameters (regressor and classifier) ')
            else:
                return cls._create_estimator_random_regressor(**params)
        elif 'classifier' in params.keys():
            return cls._create_estimator_random_classifier(**params)

    @staticmethod
    def _create_estimator_random_regressor(regressor=any_regressor('my_rgs'),
                                           preprocessing=any_preprocessing('my_pre'),
                                           max_evals=100,
                                           trial_timeout=120,
                                           seed=None,
                                           algo=tpe.suggest):
        estim = HyperoptEstimator(regressor=regressor,
                                  preprocessing=preprocessing,
                                  algo=algo,
                                  max_evals=max_evals,
                                  trial_timeout=trial_timeout,
                                  ex_preprocs=None,
                                  classifier=None,
                                  space=None,
                                  loss_fn=None,
                                  continuous_loss_fn=False,
                                  verbose=False,
                                  fit_increment=1,
                                  fit_increment_dump_filename=None,
                                  seed=seed,
                                  use_partial_fit=False,
                                  refit=True)

        return estim

    @staticmethod
    def _create_estimator_random_classifier(classifier=any_classifier('my_clf'),
                                            preprocessing=any_preprocessing('my_pre'),
                                            max_evals=100,
                                            trial_timeout=120,
                                            seed=None,
                                            algo=tpe.suggest):
        estim = HyperoptEstimator(classifier=classifier,
                                  preprocessing=preprocessing,
                                  algo=algo,
                                  max_evals=max_evals,
                                  trial_timeout=trial_timeout,
                                  ex_preprocs=None,
                                  regressor=None,
                                  space=None,
                                  loss_fn=None,
                                  continuous_loss_fn=False,
                                  verbose=False,
                                  fit_increment=1,
                                  fit_increment_dump_filename=None,
                                  seed=seed,
                                  use_partial_fit=False,
                                  refit=True)
        return estim


def test_dataset():
    return DataSetParser.iris_test_dataset()


def create_estimator(dataset_dict):
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    X_train, X_test, y_train, y_test = DataSetParser.datadict_parser(dataset_dict)

    estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                              preprocessing=any_preprocessing('my_pre'),
                              algo=tpe.suggest,
                              max_evals=100,
                              trial_timeout=120)

    # Search the hyperparameter space based on the data

    estim.fit(X_train, y_train)

    # Show the results
    estim.retrain_best_model_on_full_data(X_train, y_train)
    best_score = estim.score(X_test, y_test)
    print(best_score)

    result_dict = estim.best_model()

    result_dict['best_score'] = best_score

    return result_dict
    # 1.0


if __name__ == '__main__':
    params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 200, 'trial_timeout': 120, 'seed': None}

    params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 200, 'trial_timeout': 120,
                         'seed': None}

    print(create_estimator(test_dataset()))
