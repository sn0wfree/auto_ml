# coding=utf-8
from auto_ml.mopping.saves import saves
from auto_ml.prep.paremeter_parser import ModelStore
from auto_ml.tools.uuid_hash import uuid_hash
from auto_ml.mopping.exec_sqlite import insert_data



def create_hash(res,return_dumps=False):
    res_dumps = ModelStore._save_in_memory(res)
    model_id = uuid_hash(str(res_dumps))
    if return_dumps:
        return model_id,res_dumps
    else:
        return model_id


def save_process_docker(res,data_id,data_path,model_path,sqlite_path,log_func=None):
    if res is not None:
        model_id,res_dumps=create_hash(res,return_dumps=True)
        file_path = model_path + model_id
        saves(file_path,res_dumps)
    else:
        raise ValueError('result is None!')
    best_train_score = res['best_train_score']
    best_test_score =res['best_test_score']

    insert_data(sqlite_path,model_id,data_id,model_path,data_path,best_train_score,best_test_score,tableName='link',types='sqlite',log_func=None)

if __name__=='__main__':
    pass
