# coding=utf8
import pickle


class ParametersHolder(object):
    Regressors = ['svr', 'svr_linear', 'svr_rbf',
                  'svr_poly', 'svr_sigmoid',
                  'knn_regression', 'ada_boost_regression',
                  'gradient_boosting_regression', 'random_forest_regression', 'extra_trees_regression',
                  'sgd_regression', 'xgboost_regression']
    Classifiers = ['svc', 'svc_linear', 'svc_rbf', 'svc_poly',
                   'svc_sigmoid', 'liblinear_svc',
                   'knn', 'ada_boost', 'gradient_boosting',
                   'random_forest', 'extra_trees',
                   'decision_tree', 'sgd', 'xgboost_classification',
                   'multinomial_nb', 'gaussian_nb',
                   'passive_aggressive', 'linear_discriminant_analysis',
                   'quadratic_discriminant_analysis',
                   'one_vs_rest', 'one_vs_one', 'output_code']
    Preprocessing = ['pca', 'one_hot_encoder', 'standard_scaler',
                     'min_max_scaler', 'normalizer', 'ts_lagselector', 'tfidf',
                     'rbm', 'colkmeans']


from hpsklearn.components import xgboost, xgboost_classification, xgboost_regression
from hpsklearn.components import svc, knn, random_forest, extra_trees, ada_boost, gradient_boosting, sgd
from hpsklearn.components import svr, knn_regression, random_forest_regression, extra_trees_regression, \
    ada_boost_regression, gradient_boosting_regression, sgd_regression

from hyperopt import hp


class Parser(object):
    @staticmethod
    def _translate_obj(basic, add_extra=False, extra=['xgboot']):
        if add_extra:
            return basic + extra
        else:
            return basic

    @classmethod
    def translate_clf(cls, add_extra=False, extra=['xgboot']):
        basic = ['svc', 'knn', 'random_forest', 'extra_trees', 'ada_boost', 'gradient_boosting', 'sgd']
        return cls._translate_obj(basic, add_extra=add_extra, extra=extra)

    @classmethod
    def translate_rgs(cls, add_extra=False, extra=['xgboot']):
        basic = ['svr', 'knn', 'random_forest', 'extra_trees', 'ada_boost', 'gradient_boosting', 'sgd']
        return cls._translate_obj(basic, add_extra=add_extra, extra=extra)

    @classmethod
    def check_translate(cls, name, translate_type='rgs',printout=False):
        """

        :param name:
        :param translate_type:  rgs or clf
        :return:
        """
        if name in getattr(cls, f'translate_{translate_type}')():
            if printout:
                print(f'{name} at parameter list')
        else:
            raise ValueError(f'Unknown classifier type : {name}! ')

    @classmethod
    def select_classifier(cls, clf_type, name='clf', printout=False):

        cls.check_translate(clf_type, translate_type=name)  # check parameter
        classifiers_dict = dict(
            svc=svc(name + '.svc'),
            knn=knn(name + '.knn'),
            random_forest=random_forest(name + '.random_forest'),
            extra_trees=extra_trees(name + '.extra_trees'),
            ada_boost=ada_boost(name + '.ada_boost'),
            gradient_boosting=gradient_boosting(name + '.grad_boosting', loss='deviance'),
            sgd=sgd(name + '.sgd')
        )

        if xgboost:
            classifiers_dict['xgboost'] = xgboost_classification(name + '.xgboost')

        # if clf_type in classifiers_dict.keys():
        if printout:
            print([classifiers_dict[clf_type]])
        return hp.choice('%s' % name, [classifiers_dict[clf_type]])

        # else:
        #     raise ValueError(f'Unknown classifier type : {clf_type}! ')

    @classmethod
    def select_regressor(cls, rgs_type, name='rgs', printout=False):

        cls.check_translate(rgs_type, translate_type=name)  # check parameter

        regressors_dict = dict(
            svr=svr(name + '.svr'),
            knn=knn_regression(name + '.knn'),
            random_forest=random_forest_regression(name + '.random_forest'),
            extra_trees=extra_trees_regression(name + '.extra_trees'),
            ada_boost=ada_boost_regression(name + '.ada_boost'),
            gradient_boosting=gradient_boosting_regression(name + '.grad_boosting'),
            sgd=sgd_regression(name + '.sgd')
        )

        if xgboost:
            regressors_dict['xgboost'] = xgboost_regression(name + '.xgboost')

        # print(regressors_dict.keys())
        if printout:
            print([regressors_dict[rgs_type]])
        return hp.choice('%s' % name, [regressors_dict[rgs_type]])


class ModelStore(object):
    @staticmethod
    def grab_result(result):
        return result

    @staticmethod
    def _force_read(files_):
        """
        force read pickle object
        :param files_:
        :return:
        """
        with open(files_, 'rb') as f:
            strings = f.read()
        return strings

    @staticmethod
    def _force_read_from_string(strings):
        return pickle.loads(strings)

    @staticmethod
    def _save(modeldict, files_, protocol=2):
        with open(files_, 'wb') as f:
            pickle.dump(modeldict, f, protocol=protocol)

    @staticmethod
    def _save_in_memory(modeldict, protocol=2):
        return pickle.dumps(modeldict, protocol=protocol)

    @staticmethod
    def _read(files_):
        """

        :param files_:
        :return:
        """
        with open(files_, 'rb') as f:
            modeldict = pickle.load(f)
        return modeldict


if __name__ == '__main__':
    # params_classifier = {'classifier': None, 'preprocessing': None, 'max_evals': 5,
    #                      'trial_timeout': 100, 'seed': 1}
    # dataset_dict = test_dataset()
    # m = Models(params_classifier, dataset_dict)
    #
    # mod = m.fit_and_return(verbose_debug=False)

    # files_ = 'test.model'
    # strings = ModelStore._force_read(files_)
    #
    # M2 = ModelStore._force_read_from_string(strings)
    # print(M2)
    print(Parser.translate_rgs())
    s = Parser.select_regressor('svr')
    print(s)

    pass
