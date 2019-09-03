# coding=utf-8
import datetime
import json
import sqlite3
import pandas as pd

from auto_ml.mopping.switch_log_func import switch_log_func


def execute_sqlite_write(sqlitefile, sql, log_func=None, **kwargs):
    with sqlite3.connect(sqlitefile) as conn:
        c = conn.cusor()
        c.execute(sql)
        conn.commit()

    switch_log_func(f"{sql} SQL Command completed!", log_func=log_func)


def execute_sqlite_write(sqlitefile, sql, log_func=None, **kwargs):
    with sqlite3.connect(sqlitefile) as conn:
        df = pd.read_sql(sql, conn)

    switch_log_func(f"{sql} SQL Command completed!", log_func=log_func)
    return df


def today(fmt='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.today().strftime(fmt)


sql_insert = """INSERT INTO {table_name} (model_id,data_id,data_path,model_path,best_train_score,best_test_score,update_time) VALUES ( ({model_id},{data_id},{data_path},{model_path},{best_train_score},{best_test_score},{update_time}))"""
sql_insert_mysql = """INSERT IGNORE INTO {table_name} (model_id,data_id,data_path,model_path,best_train_score,best_test_score,update_time) VALUES ( ({model_id},{data_id},{data_path},{model_path},{best_train_score},{best_test_score},{update_time}))"""


def insert_data_sqlite(sqlfile, model_id, data_id, model_path, data_path, best_train_score, best_test_score,
                       tableName='link', types='sqlite', log_func=None):
    if types == 'sqlite':
        sql_insert2 = sql_insert
    else:
        sql_insert2 = sql_insert_mysql

    sql_comm = sql_insert2.format(table_name=tableName,
                                  model_id=json.dumps(model_id),
                                  data_id=json.dumps(data_id),
                                  data_path=json.dumps(data_path),
                                  model_path=json.dumps(model_path),
                                  best_train_score=json.dumps(best_train_score),
                                  best_test_score=json.dumps(best_test_score),
                                  update_time=json.dumps(today())
                                  )
    return execute_sqlite_write(sqlfile, sql_comm, log_func=log_func)


def insert_data(sqlfile, model_id, data_id, model_path, data_path, best_train_score, best_test_score, tableName='link',
                types='sqlite', log_func=None):
    if types == 'sqlite':
        insert_data_sqlite(sqlfile, model_id, data_id, model_path, data_path, best_train_score, best_test_score,
                           tableName=tableName, types='sqlite', log_func=log_func)
    else:
        raise ValueError('{} only support sqllite, please try again later!'.format(types))
