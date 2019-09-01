import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

train_sales  = pd.read_csv('train_sales_data.csv')
evaluation_public = pd.read_csv('evaluation_public.csv')
submit_example    = pd.read_csv('submit_example.csv')
day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
train_sales['daycount']=train_sales['regMonth'].map(day_map)
train_sales.loc[(train_sales.regMonth==2)&(train_sales.regYear==2016),'daycount']=29
train_sales['salesVolume']/=train_sales['daycount']


def rule1(train_sales):
    train_sales = train_sales.set_index(["adcode", "bodyType", "model", "province","regYear", "regMonth"])[["salesVolume"]].unstack(level=-1)
    train_sales.columns=['1','2','3','4','5','6','7','8','9','10','11','12']
    train_sales["Mean"] = train_sales.mean(axis=1)
    train_sales["Median"] = train_sales.median(axis=1)

    train_sales['quarter1_mean']=(train_sales['1']+train_sales['2']+train_sales['3'])/3
    train_sales['quarter2_mean']=(train_sales['4']+train_sales['5']+train_sales['6'])/3

    train_sales=train_sales.unstack("regYear")
    train_sales.columns= ["_".join([str(i) for i in x]) for x in train_sales.columns.ravel()]
    train_sales = train_sales.reset_index()
    train_sales['20171_divide_20161']=train_sales['1_2017']/train_sales['1_2016']
    train_sales['20172_divide_20162']=train_sales['2_2017']/train_sales['2_2016']
    train_sales['20173_divide_20163']=train_sales['3_2017']/train_sales['3_2016']
    train_sales['20174_divide_20164']=train_sales['4_2017']/train_sales['4_2016']

    train_sales['2017y_divide_2016y_mean']=train_sales['Mean_2017']/train_sales['Mean_2016']
    train_sales['2017y_divide_2016y_median']=train_sales['Median_2017']/train_sales['Median_2016']
    train_sales['2017q1_divide_2016q1']=train_sales['quarter1_mean_2017']/train_sales['quarter1_mean_2016']
    train_sales['2017q2_divide_2016q2']=train_sales['quarter2_mean_2017']/train_sales['quarter2_mean_2016']

    qushim1=(0.25*train_sales['2017y_divide_2016y_mean']+0.25*train_sales['2017y_divide_2016y_mean'])+0.3*train_sales['2017q1_divide_2016q1']+0.2*train_sales['20171_divide_20161']
    qushim2=(0.25*train_sales['2017y_divide_2016y_mean']+0.25*train_sales['2017y_divide_2016y_mean'])+0.3*train_sales['2017q1_divide_2016q1']+0.2*train_sales['20172_divide_20162']
    qushim3=(0.25*train_sales['2017y_divide_2016y_mean']+0.25*train_sales['2017y_divide_2016y_mean'])+0.3*train_sales['2017q1_divide_2016q1']+0.2*train_sales['20173_divide_20163']
    qushim4=(0.25*train_sales['2017y_divide_2016y_mean']+0.25*train_sales['2017y_divide_2016y_mean'])+0.3*train_sales['2017q2_divide_2016q2']+0.2*train_sales['20174_divide_20164']


    qushi1_v_2016=train_sales['1_2016'].values* qushim1* qushim1
    qushi2_v_2016=train_sales['2_2016'].values* qushim2* qushim2
    qushi3_v_2016=train_sales['3_2016'].values* qushim3* qushim3
    qushi4_v_2016=train_sales['4_2016'].values* qushim4* qushim4

    qushi1_v_2017=train_sales['1_2017'].values* qushim1
    qushi2_v_2017=train_sales['2_2017'].values* qushim2
    qushi3_v_2017=train_sales['3_2017'].values* qushim3
    qushi4_v_2017=train_sales['4_2017'].values* qushim4

    qushi1_v=qushi1_v_2016*0.3+qushi1_v_2017*0.7
    qushi2_v=qushi2_v_2016*0.3+qushi2_v_2017*0.7
    qushi3_v=qushi3_v_2016*0.3+qushi3_v_2017*0.7
    qushi4_v=qushi4_v_2016*0.3+qushi4_v_2017*0.7

    train_sales['pre1']=qushi1_v
    train_sales['pre2']=qushi2_v
    train_sales['pre3']=qushi3_v
    train_sales['pre4']=qushi4_v
    presale1 = train_sales[["adcode", "bodyType", "model", "province",'pre1']].rename(columns={'pre1':'forecastVolum1'})
    presale1['regMonth']=1
    presale2 = train_sales[["adcode", "bodyType", "model", "province", 'pre2']].rename(columns={'pre2':'forecastVolum1'})
    presale2['regMonth'] = 2
    presale3 = train_sales[["adcode", "bodyType", "model", "province", 'pre3']].rename(columns={'pre3':'forecastVolum1'})
    presale3['regMonth'] = 3
    presale4 = train_sales[["adcode", "bodyType", "model", "province", 'pre4']].rename(columns={'pre4':'forecastVolum1'})
    presale4['regMonth'] = 4

    result=pd.concat([presale1,presale2,presale3,presale4]).reset_index(drop=True)

    return result

rule1=rule1(train_sales.copy())

def yulaorule(train_sales):
    m1_12 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
            train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 12), 'salesVolume'].values
    m1_11 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
            train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 11), 'salesVolume'].values
    m1_10 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
            train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 10), 'salesVolume'].values
    m1_09 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
            train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 9), 'salesVolume'].values
    m1_08 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
            train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 8), 'salesVolume'].values

    m1_12_volum = train_sales.loc[
                      (train_sales.regYear == 2017) & (train_sales.regMonth == 12), 'salesVolume'].values * m1_12
    m1_11_volum = train_sales.loc[
                      (train_sales.regYear == 2017) & (train_sales.regMonth == 11), 'salesVolume'].values * m1_11
    m1_10_volum = train_sales.loc[
                      (train_sales.regYear == 2017) & (train_sales.regMonth == 10), 'salesVolume'].values * m1_10
    m1_09_volum = train_sales.loc[
                      (train_sales.regYear == 2017) & (train_sales.regMonth == 9), 'salesVolume'].values * m1_09
    m1_08_volum = train_sales.loc[
                      (train_sales.regYear == 2017) & (train_sales.regMonth == 8), 'salesVolume'].values * m1_08

    evaluation_public.loc[
        evaluation_public.regMonth == 1, 'forecastVolum'] = m1_12_volum / 2 + m1_11_volum / 4 + m1_10_volum / 8 + m1_09_volum / 16 + m1_08_volum / 16

    # 2018年1、2、3月，提取方式历史月份销量比例，考虑时间衰减，月份越近占比越高
    m16_1_2 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values
    m16_1_3 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values
    m16_1_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
    m16_1_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values

    m16_2_3 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values
    m16_2_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
    m16_2_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values
    m16_2_6 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 6), 'salesVolume'].values

    m16_3_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
    m16_3_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values
    m16_3_6 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 6), 'salesVolume'].values
    m16_3_7 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 7), 'salesVolume'].values

    m17_1_2 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values
    m17_1_3 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values
    m17_1_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
    m17_1_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values

    m17_2_3 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values
    m17_2_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
    m17_2_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values
    m17_2_6 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 6), 'salesVolume'].values

    m17_3_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
    m17_3_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values
    m17_3_6 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 6), 'salesVolume'].values
    m17_3_7 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
              train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 7), 'salesVolume'].values

    m16_1 = m16_1_2 / 2 + m16_1_3 / 4 + m16_1_4 / 8 + m16_1_5 / 8
    m16_2 = m16_2_3 / 2 + m16_2_4 / 4 + m16_2_5 / 8 + m16_2_6 / 8
    m16_3 = m16_3_4 / 2 + m16_3_5 / 4 + m16_3_6 / 8 + m16_3_7 / 8

    m17_1 = m17_1_2 / 2 + m17_1_3 / 4 + m17_1_4 / 8 + m17_1_5 / 8
    m17_2 = m17_2_3 / 2 + m17_2_4 / 4 + m17_2_5 / 8 + m17_2_6 / 8
    m17_3 = m17_3_4 / 2 + m17_3_5 / 4 + m17_3_6 / 8 + m17_3_7 / 8

    m1 = m16_1 * 0.4 + m17_1 * 0.6
    m2 = m16_2 * 0.4 + m17_2 * 0.6
    m3 = m16_3 * 0.4 + m17_3 * 0.6

    evaluation_public.loc[evaluation_public.regMonth == 2, 'forecastVolum'] = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / m1
    evaluation_public.loc[evaluation_public.regMonth == 3, 'forecastVolum'] = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / m2
    evaluation_public.loc[evaluation_public.regMonth == 4, 'forecastVolum'] = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / m3
    return evaluation_public

rule2=yulaorule(train_sales.copy())

rule=rule2.merge(rule1,on=["adcode", "model", "province","regMonth"],how='left').rename(columns={'forecastVolum':'forecastVolum2'})
day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
rule['daycount']=rule['regMonth'].map(day_map)
rule['forecastVolum']=(0.5*rule['forecastVolum2']+0.5*rule['forecastVolum1'])*rule['daycount']
