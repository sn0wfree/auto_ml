#coding=utf8
from auto_ml.mopping.switch_log_func import switch_log_func
from auto_ml.tools.retry_it import retry

@retry(max_retries=5, default_retry_delay=1, default_sleep_time=0.1, Exception_func=Exception)
def saves(file_path,res_dumps,log_func=None):
    msg = f'result dict has been save at {file_path}'
    with open(file_path,'wb') as f:
        f.write(res_dumps)
    switch_log_func(msg,log_func=log_func)