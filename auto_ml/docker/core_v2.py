# coding=utf8

import sys
from collections import ChainMap

from hpsklearn import HyperoptEstimator, any_classifier, any_regressor

from hyperopt import tpe

from auto_ml.mopping.switch_log_func import switch_log_func
from auto_ml.prep.costum_error import ParametersError

from auto_ml.prep.paremeter_parser import Parser, ParametersHolder, ModelStore
from auto_ml.tools.uuid_hash import uuid_hash

Regressors = ParametersHolder.Regressors
Classifiers = ParametersHolder.Classifiers
Preprocessing = ParametersHolder.Preprocessing

search_mode_dict_rgs = {rgs: 'rgs' for rgs in Regressors}
search_mode_dict_clf = {clf: 'clf' for clf in Classifiers}
ext_dict = {'rgs': 'rgs', 'clf': 'clf'}
all_dict = ChainMap(search_mode_dict_rgs, search_mode_dict_clf, ext_dict)


# search_mode_dict_rgs = {rgs:'rgs' for rgs in Regressors}


def custom_core(name, core_type):
    if core_type == 'clf':
        if name == 'clf':
            return any_classifier(name)
        else:
            return Parser.select_classifier(name)
    elif core_type == 'rgs':
        if name == 'rgs':
            return any_regressor(name)
        else:
            return Parser.select_classifier(name)
    else:
        raise ValueError(f'unknown core_type: {core_type}')


def _create_core(dataset_dict, max_evals, trial_timeout, method=(None, None), preprocessing=[],
                 ex_preprocs=None,
                 space=None,
                 loss_fn=None,
                 verbose=False,
                 fit_increment=1,
                 fit_increment_dump_filename=None,
                 seed=None,
                 use_partial_fit=False,
                 ):
    regressor, classifier = method

    if regressor is not None and classifier is not None:
        raise ParametersError(
            'regressor and classifier both are not NOne, only receive one type, please set one of them as None!')
    elif regressor is None and classifier is None:
        switch_log_func('ad', log_func=None)
        status = 'regressor'
        regressor = 'my_rgs'
        unit = any_regressor(regressor)

    elif regressor is None and classifier is not None:
        status = 'classifier'
        if classifier == 0:
            classifier = 'my_clf'
            unit = any_classifier(classifier)
        else:
            unit = custom_core(classifier, core_type='clf')
    else:
        status = 'regressor'
        if regressor == 0:
            regressor = 'my_rgs'
            unit = any_regressor(regressor)
        else:
            unit = custom_core(regressor, core_type='rgs')

    if status == 'regressor':
        estim = HyperoptEstimator(regressor=unit,
                                  preprocessing=preprocessing,
                                  algo=tpe.suggest,
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
    else:
        estim = HyperoptEstimator(classifier=unit,
                                  preprocessing=preprocessing,
                                  algo=tpe.suggest,
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


def run_core(dataset_dict, max_evals, trial_timeout, method=(None, None), preprocessing=[],
             ex_preprocs=None,
             space=None,
             loss_fn=None,
             verbose=False,
             fit_increment=1,
             fit_increment_dump_filename=None,
             seed=None,
             use_partial_fit=False, raiseError=False):
    core = _create_core(dataset_dict, max_evals, trial_timeout, method=method, preprocessing=preprocessing,
                        ex_preprocs=ex_preprocs,
                        space=space,
                        loss_fn=loss_fn,
                        verbose=verbose,
                        fit_increment=fit_increment,
                        fit_increment_dump_filename=fit_increment_dump_filename,
                        seed=seed,
                        use_partial_fit=use_partial_fit)

    try:
        core.fit(dataset_dict['X_train'], dataset_dict['y_train'])
    except Exception as e:
        if raiseError:
            return repr(e)
        else:
            print(e)
            return None
    else:
        best_train_score = core.score(dataset_dict['X_train'], dataset_dict['y_train'])
        best_test_score = core.score(dataset_dict['X_test'], dataset_dict['y_test'])
        print(f'best_train_score: {best_train_score}; \n best_test_score: {best_test_score}')
        return {'model': core.best_model(), 'best_train_score': best_train_score, 'best_test_score': best_test_score}


def create_hash(res):
    res_dumps = ModelStore._save_in_memory(res)
    model_id = hash(str(res_dumps))
    return model_id
