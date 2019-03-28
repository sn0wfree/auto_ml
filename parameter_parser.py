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


class Parser(object):
    pass


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
            strings = f.readlines()
        return strings

    @staticmethod
    def _force_read_from_string(strings):
        return pickle.loads(strings)

    @staticmethod
    def _save(modeldict, files_, protocol=2):
        with open(files_, 'wb') as f:
            pickle.dump(modeldict, f, protocol=protocol)

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
    print(ParametersHolder.Regressors)
    pass
