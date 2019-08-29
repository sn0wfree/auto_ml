# coding=utf8
import time
from functools import wraps


def conn_try_again(max_retries=5, default_retry_delay=1, Exception_func=Exception):
    """
    retry function
    :param max_retries:
    :param default_retry_delay:
    :return:
    """

    def _conn_try_again(function):
        RETRIES = 0
        # 重试的次数
        count = {"num": RETRIES}

        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception_func as err:
                print(err)

                if count['num'] < max_retries:
                    print('will retry {} times '.format(max_retries - count['num']))
                    time.sleep(default_retry_delay)
                    count['num'] += 1
                    return wrapped(*args, **kwargs)
                else:
                    # status = 'Error'
                    # sel = 'Error'
                    raise Exception(err)

        return wrapped

    return _conn_try_again


if __name__ == '__main__':
    pass
