# coding:utf-8
from auto_ml.core.mini_bridge_v2 import elf_func
from auto_ml.tools.typeassert import typeassert


class AutoMLExt(object):
    @staticmethod
    def main_enter(model_store_path, data_path, max_evals, trial_timeout, search_mode, sqlfile_path,
                   eval_func=lambda x: True, yieldout=True):
        return elf_func(model_store_path, data_path, max_evals, trial_timeout, search_mode, sqlfile_path,
                        eval_func=eval_func, yieldout=yieldout)


if __name__ == '__main__':
    pass
