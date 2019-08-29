#coding=utf8

def switch_log_func(msg,log_func=None):
    if log_func is None:
        print(msg)
    else:
        try:
            log_func(msg)

        except Exception as e:
            print(e)
            print(msg)
            print('force into print func to print')
        else:
            pass