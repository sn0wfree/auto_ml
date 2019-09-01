# coding=utf-8
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing, any_regressor

from hyperopt import tpe
import numpy as np
# from auto_ml.tools.conn_try_again import conn_try_again
# from auto_ml.tools.typeassert import typeassert
import copy
# from auto_ml.core.parameter_parser import Parser
max_retries = 5
default_retry_delay = 1


class ModelBuilder(object):
    param_regressor = ['regressor','preprocessing',  'max_evals', 'trial_timeout']
    param_classifier = ['classifier','preprocessing', 'max_evals', 'trial_timeout']

    @classmethod
    def _create_estimator(cls, **kwargs):

        estim = HyperoptEstimator(**kwargs)

        return estim

    @classmethod
#     @typeassert(object, params=dict)
    def create_estimator(cls, params, param_type='rgs'):
        if params['preprocessing'] is None:
            params['preprocessing']=any_preprocessing('my_pre')
        if param_type == 'rgs':
            if params['regressor'] is None:
                params['regressor']=any_regressor('my_rgs')
            return cls._create_estimator_random_regressor(**params)
        elif param_type == 'clf':
            if params['classifier'] is None:
                params['classifier']=any_classifier('my_clf')
            return cls._create_estimator_random_classifier(**params)
        else:
            raise ValueError('wrong param_type!')
            
       

    @staticmethod
    def _create_estimator_random_regressor(regressor=any_regressor('my_rgs'),
                                           preprocessing=any_preprocessing('my_pre'),
                                           max_evals=100,
                                           trial_timeout=120,
                                           seed=None,
                                           algo=tpe.suggest
                                           ):
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



