# coding=utf-8
import os
from auto_ml.api._api import ModelStore
from auto_ml.mopping.save_process_for_docker import save_process_docker
from auto_ml.core.core_v2 import load_and_run_ml, all_dict
from auto_ml.prep.costum_error import NoResultError, RetryError
from auto_ml.tools import config_for_all

from auto_ml.tools.retry_it import retry

config_for_all.auto(_config_file='config_for_all.conf')

inner_failure_retry_limit = config_for_all.ALL['inner_failure_retry_limit']
total_failure_retry_limit = config_for_all.ALL['total_failure_retry_limit']


@retry(max_retries=inner_failure_retry_limit, default_retry_delay=0.01, default_sleep_time=0.01,
       Exception_func=NoResultError)
def auto_run_core(data_dict, max_evals, trial_timeout, search_mode):
    result_dict = load_and_run_ml(data_dict, max_evals, trial_timeout, search_mode, core_reflect_dict=all_dict)

    if result_dict is None:
        raise NoResultError('there is no result!')
    else:
        return result_dict


@retry(max_retries=total_failure_retry_limit, default_retry_delay=0.01, default_sleep_time=0.01,
       Exception_func=RetryError)
def auto_run(model_path, data_path, max_evals, trial_timeout, search_mode, eval_func=lambda x: True):
    dataid = data_path
    dataset_dict = ModelStore._read(dataid)

    result = load_and_run_ml(dataset_dict, max_evals, trial_timeout, search_mode)

    if eval_func(result):
        raise RetryError('retry it!')
    else:
        pass
    return result


def elf_func(model_store_path,data_path, max_evals, trial_timeout, search_mode, sqlfile_path,eval_func=lambda x: True,yieldout=True):
    result = auto_run(model_store_path,data_path, max_evals, trial_timeout, search_mode,eval_func=eval_func)
    data_read_path,data_id = os.path.split(data_path)
    model_path = model_store_path
    sqlfile_path = sqlfile_path
    save_process_docker(result,data_id,data_read_path,model_path,sqlite_path=sqlfile_path,log_func=None)


    if yieldout:
        return result
    else:
        pass
