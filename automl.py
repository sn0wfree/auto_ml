# coding=utf8

from auto_ml_core import test_dataset, Models
import pickle
from parameter_parser import ModelStore


def test(params):
    # params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 15,
    #                     'trial_timeout': 100, 'seed': 1}
    #
    # params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 15,
    #                      'trial_timeout': 100, 'seed': 1}

    # estimator = ModelBuilder.create_estimator(params_regressor)
    dataset_dict = test_dataset()
    m = Models(params, dataset_dict)

    return m.fit_and_return(verbose_debug=False)


if __name__ == '__main__':
    params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 5,
                        'trial_timeout': 100, 'seed': 1}

    params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 5,
                         'trial_timeout': 100, 'seed': 1}

    # estimator = ModelBuilder.create_estimator(params_regressor)
    modeldict = test(params_regressor)
    files_ = 'test.model'
    ModelStore._save(modeldict, files_)
    modeldict2 = ModelStore._read(files_, protocol=2)

    print(modeldict, modeldict2)
