# coding=utf8
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing, any_regressor

from hyperopt import tpe
import numpy as np
from auto_ml.tools.retry_it import retry
from auto_ml.tools.typeassert import typeassert
import copy

from auto_ml.core.parameter_parser import Parser

max_retries = 5
default_retry_delay = 1


class DataSetParser(object):

    @staticmethod
    @typeassert(dataset_dict=dict)
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
    param_regressor = ['regressor', 'preprocessing', 'algo', 'max_evals', 'trial_timeout', 'seed']
    param_classifier = ['classifier', 'preprocessing', 'algo', 'max_evals', 'trial_timeout', 'seed']

    @classmethod
    def _create_estimator(cls, **kwargs):
        # dict(regressor=regressor,
        #      preprocessing=preprocessing,
        #      algo=algo,
        #      max_evals=max_evals,
        #      trial_timeout=trial_timeout,
        #      seed=seed)

        # params = dict(regressor=None,
        #          preprocessing=None,
        #          algo=None,
        #          max_evals=None,
        #          trial_timeout=None,
        #          ex_preprocs=None,
        #          classifier=None,
        #          space=None,
        #          loss_fn=None,
        #          continuous_loss_fn=False,
        #          verbose=False,
        #          fit_increment=1,
        #          fit_increment_dump_filename=None,
        #          seed=None,
        #          use_partial_fit=False,
        #          refit=True)
        estim = HyperoptEstimator(**kwargs)

        return estim

    @classmethod
    @typeassert(object, params=dict)
    def create_estimator(cls, params, printout=False):
        """



        :param params:
        :return:
        """
        if 'regressor' in params.keys():
            if 'classifier' in params.keys():
                raise ValueError('params obtain two similar parameters (regressor and classifier) ')
            else:
                params_copy = copy.deepcopy(params)
                if printout:
                    print('regressor', params_copy)
                if params_copy['regressor'] is None:
                    params_copy.pop('regressor')

                    return cls._create_estimator_random_regressor(**params_copy)
                else:

                    params_copy['regressor'] = Parser.select_regressor(params_copy['regressor'])
                    if printout:
                        print(params_copy['regressor'])
                    return cls._create_estimator(**params_copy)

        elif 'classifier' in params.keys():

            params_copy = copy.deepcopy(params)
            if printout:
                print('classifier', params_copy)

            if params_copy['classifier'] is None:
                params_copy.pop('classifier')

                return cls._create_estimator_random_classifier(**params_copy)
            else:
                # print(Parser.select_classifier(params['classifier']))
                params_copy['classifier'] = Parser.select_classifier(params_copy['classifier'])
                if printout:
                    print(params_copy['classifier'])
                return cls._create_estimator(**params_copy)

    @staticmethod
    def _create_estimator_random_regressor(regressor=any_regressor('my_rgs'),
                                           preprocessing=any_preprocessing('my_pre'),
                                           max_evals=100,
                                           trial_timeout=120,
                                           seed=None,
                                           algo=tpe.suggest, fit_increment=1):
        """

        :param regressor:
        :param preprocessing:
        :param max_evals:
        :param trial_timeout:
        :param seed:
        :param algo:
        :return:
        """

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
                                  fit_increment=fit_increment,
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
        """

        :param classifier:
        :param preprocessing:
        :param max_evals:
        :param trial_timeout:
        :param seed:
        :param algo:
        :return:
        """
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


class Model(object):
    @typeassert(object, estimator=object, dataset_dict=dict)
    def __init__(self, estimator, dataset_dict):
        # self._ModelBuilder = ModelBuilder

        self.dataset_dict = dataset_dict
        self.estimator = estimator
        self.result = {}
        self.result_verbose = {}

    @conn_try_again(max_retries=max_retries, default_retry_delay=default_retry_delay)
    def _fit(self, X_train, y_train, verbose_debug=False):
        if verbose_debug:
            print('fit_iter')
            iterator = self.estimator.fit_iter(X_train, y_train)
        else:
            iterator = self.estimator.fit(X_train, y_train)
        return iterator

    def fit(self, verbose_debug=False):
        """
        because of some model can be fit with given data, thus this fit function is unsafe function
        :param verbose_debug:
        :return:
        """
        X_train = self.dataset_dict['X_train']
        y_train = self.dataset_dict['y_train']
        return self._fit(X_train, y_train, verbose_debug=verbose_debug)

    @retry(max_retries=max_retries, default_retry_delay=default_retry_delay)
    def fit_and_return(self, verbose_debug=False):
        iterator = self.fit(verbose_debug=verbose_debug)
        if verbose_debug:
            n_trails = 0
            for model in iterator:
                # X_train = self.dataset_dict['X_train']
                # y_train = self.dataset_dict['y_train']
                #
                # X_test = self.dataset_dict['X_test']
                # y_test = self.dataset_dict['y_test']
                train_score = self.train_score
                test_score = self.test_score
                print('Trails {} | Training Score : {} | Testing Score : {}'.format(n_trails, train_score, test_score))

                n_trails += 1
                self.result = self.best_model()
                self.result['best_test_score'] = self.best_test_score

                self.result['train_score'] = train_score
                self.result['test_score'] = test_score

                self.result_verbose['{}'.format(n_trails)] = self.result

                return self.result_verbose

        else:

            self.result = self.best_model()

            self.result['best_train_score'] = self.train_score
            self.result['best_test_score'] = self.best_test_score

            return self.result

    @property
    def train_score(self):
        X_train = self.dataset_dict['X_train']
        y_train = self.dataset_dict['y_train']
        return self.estimator.score(X_train, y_train)

    @property
    def test_score(self):
        X_test = self.dataset_dict['X_test']
        y_test = self.dataset_dict['y_test']

        return self.estimator.score(X_test, y_test)

    @property
    def best_train_score(self):
        return self.train_score

    @property
    def best_test_score(self):
        return self.test_score

    # def _best_score(self, X_test, y_test):
    #     return self.estimator.score(X_test, y_test)

    def retrain_best_model_on_full_data(self, X_train, y_train):
        return self.estimator.retrain_best_model_on_full_data(X_train, y_train)

    def best_model(self):
        return self.estimator.best_model()


class Models(object):

    @typeassert(object, dict, dict)
    def __init__(self, params, datatset):
        self.params = params
        self.datatset = datatset
        self._model = None

    def fit_and_return(self, verbose_debug=False):
        estimator = self._create_estimator()
        self._model = Model(estimator, self.datatset)
        return self._model.fit_and_return(verbose_debug=verbose_debug)

    @typeassert(object, int)
    def _create_estimator(self, multi=None):
        if multi is None:
            estimator = ModelBuilder.create_estimator(self.params)
            return estimator
        else:
            raise ValueError('incomplete part!')
            return [self._ModelBuilder.create_estimator(self.params) for _ in range(multi)]


def test_dataset():
    return DataSetParser.iris_test_dataset()


# def create_estimator(dataset_dict):
#     # Instantiate a HyperoptEstimator with the search space and number of evaluations
#     X_train, X_test, y_train, y_test = DataSetParser.datadict_parser(dataset_dict)
#
#     estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
#                               preprocessing=any_preprocessing('my_pre'),
#                               algo=tpe.suggest,
#                               max_evals=100,
#                               trial_timeout=120)
#
#     # Search the hyperparameter space based on the data
#
#     estim.fit(X_train, y_train)
#
#     # Show the results
#     estim.retrain_best_model_on_full_data(X_train, y_train)
#     best_score = estim.score(X_test, y_test)
#     print(best_score)
#
#     result_dict = estim.best_model()
#
#     result_dict['best_score'] = best_score
#
#     return result_dict
#     # 1.0


if __name__ == '__main__':
    from hpsklearn.components import random_forest

    s = random_forest('clf' + '.random_forest'),
    params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 15,
                        'trial_timeout': 100, 'seed': 1}

    params_classifier = {'classifier': s, 'preprocessing': None, 'max_evals': 15,
                         'trial_timeout': 100, 'seed': 1}

    s2 = any_classifier('te')
    print(1)

    # estimator = ModelBuilder.create_estimator(params_regressor)

    # dataset_dict = test_dataset()
    # m = Models(params_classifier, dataset_dict)
    #
    # print(m.fit_and_return(verbose_debug=False))

    print(0)
    # print(create_estimator(test_dataset()))
