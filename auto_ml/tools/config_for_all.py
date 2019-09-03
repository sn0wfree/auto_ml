# coding=utf-8
import configparser
import os
from glob import glob
from platform import system

_conf_file = 'config_for_all.conf'


def get_real_path(conf_file='config_for_all.conf'):
    if system() == 'Windows':
        sep = '\\'
    else:
        sep = '/'
    path1 = os.path.abspath('.')
    father_path = os.path.abspath(os.path.dirname(os.getcwd()))
    superfather_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    print(path1)
    if path1 + sep + conf_file in glob(path1 + sep + '*'):
        return path1 + sep
    elif father_path + sep + conf_file in glob(path1 + sep + '*'):
        return father_path + sep
    elif superfather_path + sep + conf_file in glob(path1 + sep + '*'):
        return superfather_path + sep
    else:
        raise FileNotFoundError(f'cannot locate {conf_file} at {__file__} and below 3 level folder')


class _CommonConfig(object):
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        with open(config_file, 'r') as configfile:
            self.config.readfp(configfile)
        self.sections = self.config.sections()

    def generate_config(self, cls, name, dtype):
        if dtype == 'int':
            try:
                return self.config.getint(cls, name)
            except ValueError as e:
                if 'invalid literal for int() with base 10' in str(e):
                    import warnings
                    warnings.warn("require int type data, receive other type! will transfer to float dtype!")
                    return self.config.getfloat(cls, name)
                else:
                    return self.config.getint(cls, name)
        elif dtype == 'float':
            return self.config.getfloat(cls, name)
        elif dtype == 'bool':
            return self.config.getboolean(cls, name)
        else:
            string = self.config.get(cls, name)
            if ',' in string:
                return string.split(',')
            else:
                return string

    @staticmethod
    def finder_type(v):
        if isinstance(v, str) and v.isdigit():
            try:
                return int(v)
            except ValueError as e:
                if 'invalid literal for int() with base 10' in str(e):
                    import warnings
                    warnings.warn("require int type data, receive other type! will transfer to float dtype!")
                    return float(v)
                else:
                    return v
            except Exception as e:
                import warnings
                warnings.warn(e)
                return v
        else:
            return v

    def parse_general(self, section, auto=True):
        options = self.config.options(section)
        return {option: self.finder_type(self.generate_config(section, option, 'str')) for option in options}

    def run_all(self):
        for section in self.sections:
            yield section, self.parse_general(section)


def get_configs(config_file):
    CC = _CommonConfig(config_file)
    for section_name, sec in CC.run_all():
        globals()["{}".format(section_name)] = sec


def auto(_config_file=_conf_file):
    conf_file = get_real_path(conf_file=_conf_file) + _conf_file
    print("{}:".format(_conf_file), conf_file)
    get_configs(conf_file)


if __name__ == '__main__':
    pass
