# coding=utf8

from core.auto_ml_core import test_dataset, Models
from core.parameter_parser import ModelStore


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
    params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 15,
                        'trial_timeout': 100, 'seed': 1}

    params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 15,
                         'trial_timeout': 100, 'seed': 1}

    # estimator = ModelBuilder.create_estimator(params_regressor)
    dataset_dict = test_dataset()
    m = Models(params_classifier, dataset_dict)

    print(m.fit_and_return(verbose_debug=False))



