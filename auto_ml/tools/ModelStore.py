#coding=utf-8

import pickle

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
