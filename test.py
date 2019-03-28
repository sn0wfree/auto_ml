# coding=utf8
import requests
from parameter_parser import ModelStore


def test_upload_file(files_='test.model'):
    strings = ModelStore._force_read(files_)

    # "text/plain"
    data = {'file': ('model.model', strings, "application/octet-stream")}
    # M2 = ModelStore._force_read_from_string(strings)

    r = requests.post('http://0.0.0.0:8279/upload_file', files=data)

    print(r.text)


def test_auto_ml():
    params_regressor ={'regressor': 'Null', 'preprocessing': 'Null', 'max_evals': 5,
                        'trial_timeout': 100, 'seed': 1}
    para = {'parameters': {'regressor': 'Null', 'preprocessing': 'Null', 'max_evals': 5,
                        'trial_timeout': 100, 'seed': 1}}

    r = requests.post('http://0.0.0.0:8279/auto_ml', files=para)


if __name__ == '__main__':
    test_auto_ml()
