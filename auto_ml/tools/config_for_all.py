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