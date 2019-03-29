# coding=utf8
import warnings
from core.auto_ml_core import test_dataset, Models
from core.parameter_parser import ModelStore, Parser


class GetSupportModels(object):
    @staticmethod
    def get_clf(printout=False):
        th = Parser.translate_clf()
        if printout:
            print('clf', th)
        return th

    @staticmethod
    def get_rgs(printout=False):
        th = Parser.translate_rgs()
        if printout:
            print('rgs', th)
        return th


class AML(object):
    @staticmethod
    def t():
        print('good')

    @staticmethod
    def get_supported_model(model_type='rgs', raiseError=True):
        if model_type in ['rgs', 'clf']:
            return getattr(GetSupportModels, f'get_{model_type}')()
        else:
            st = f'wrong paramter : {model_type}! rgs or clf required! '
            if raiseError:
                raise ValueError(st)
            else:
                warnings.warn(st)
                return st

    @staticmethod
    def run(params, dataset_dict):
        m = Models(params, dataset_dict)
        return m.fit_and_return(verbose_debug=False)


def test(params):
    # params_regressor = {'regressor': None, 'preprocessing': None, 'max_evals': 15,
    #                     'trial_timeout': 100, 'seed': 1}
    #
    # params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 15,
    #                      'trial_timeout': 100, 'seed': 1}

    # estimator = ModelBuilder.create_estimator(params_regressor)
    dataset_dict = test_dataset()

    return AML.run(params, dataset_dict)


if __name__ == '__main__':
    print('clf', GetSupportModels.get_clf())

    print('rgs', GetSupportModels.get_rgs())

    params_regressor = {'regressor': GetSupportModels.get_rgs()[0], 'preprocessing': [], 'max_evals': 15,
                        'trial_timeout': 100, 'seed': 1}

    params_classifier = {'classifier': GetSupportModels.get_clf()[0], 'preprocessing': [], 'max_evals': 15,
                         'trial_timeout': 100, 'seed': 1}

    # estimator = ModelBuilder.create_estimator(params_regressor)

    print(test(params_classifier))
