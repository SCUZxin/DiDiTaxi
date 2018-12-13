# proportion matrix based prediction model 基于比例矩阵的预测模型
# 先不用全部预测，只是简单预测几个时间片试一下效果

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle

import extract_feature.gen_proportion as gen_proportion


# 获取所有训练数据和测试数据，新建一列 counter（1-1152）由 day + time 组合而成
# 包含所有数据的总流量1152，比例矩阵1152个，每个都是59*59，特征向量->特征相似度（实时计算）
def get_all_data():
    # 读取测试集并加入counter
    df_test = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv')
    df_test['counter'] = (df_test['day']-1) * 48 + df_test['time'] + 1

    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\before\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    del df_time['date']
    del df_time['time']
    df = pd.concat([df_weather, df_time], axis=1)
    df['counter'] = range(1, 1153)
    flow_total = list(df['count'].values)
    proportion_list = gen_proportion.get_proportion()
    return df, flow_total, proportion_list, df_test


# 单个时间片的预测，第 t (1-1152) 个时间片的订单量，需要前面 1-H ， 此处先用3个时间片
def predict_single(t=817, len_pre_t=3):
    global flow_total
    # 需要的前 1-H 个时间片 closeness
    closeness_time_list = [i for i in range(t-len_pre_t, t)]

    # 上几周该时间片 period
    period_time_list = []
    ind = t
    while ind > 7*48:
        ind = ind - 7*48
        period_time_list.append(ind)

    # w_closeness = []
    # prop_closeness = []
    sum_w_closeness = 0
    sum_prop_closeness = np.zeros((59, 59))
    # closeness 计算这些时间片的特征相似性
    for t1 in closeness_time_list:
        sum_w_closeness += w_features(t1, t)
        sum_prop_closeness += proportion_list[t1-1]
        # w_closeness.append(w_features(t1, t))
        # prop_closeness.append(proportion_list[t1-1])

    # period 计算这些时间片的特征相似性

    # 加权平均
    prop_t = sum_prop_closeness / sum_w_closeness
    # 流量矩阵 = 总流量 * prop_t，得到59*59的数组
    flow_pair_t = flow_total[t-1] * prop_t
    return flow_pair_t


# 批量预测，预测多个时间片的流量
def predict_batch(list_t):
    flow_pair_batch_list = []
    df_target = pd.DataFrame()
    for t in list_t:
        flow_pair_batch_list.append(predict_single(t, len_pre_t=3))

    global df_test
    for i in range(len(list_t)):
        df_tmp = df_test.loc[df_test['counter'] == list_t[i]]
        df_target = pd.concat([df_target, df_tmp]).reset_index(drop=True)

    metrics_self(flow_pair_batch_list, df_target, list_t)


# flow_pair_batch_list：按照时间 list_t 装多个list，每个list是对应时间片的订单量的数组 59*59
def metrics_self(flow_pair_batch_list, df_target, list_t):
    y_test = []
    y_predict = []
    for i in range(len(list_t)):
        flow_pair_t = flow_pair_batch_list[i]
        counter = list_t[i]
        df_tmp = df_target.loc[df_target['counter'] == list_t[i]].reset_index(drop=True)
        y_test.extend(df_tmp['count'].values)
        for j in range(len(df_tmp)):
            s = df_tmp.loc[j, 'start_district_id']
            d = df_tmp.loc[j, 'dest_district_id']
            y_predict.append(flow_pair_batch_list[i][s][d])

    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(y_test, y_predict)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好



def w_features(t1, t2):
    w_t1_t2 = lambda_time(t1, t2) * lambda_weather(t1, t2)
    return w_t1_t2


# 计算时间的相似性, t: 1-1152
def lambda_time(t1, t2, rho_1=0.9, rho_2=0.9):
    global df
    period = True
    # closeness,，时间片很接近
    if abs(t2 - t1) <= 20:
        period = False
    # 只要不是同是工作日/周末，不同是假日/非假日，lambda_1 = 0
    # 这个只对period生效，closeness 如何衡量
    if period:
        if df.loc[t1-1, 'is_weekend'] != df.loc[t2-1, 'is_weekend']:
            return 0
        if df.loc[t1-1, 'is_vocation'] != df.loc[t2-1, 'is_vocation']:
            return 0

    d_t1_t2 = abs(t1 - t2) % 48
    delta_h = min(d_t1_t2, 48-d_t1_t2)
    # print(delta_h)
    # delta_d = math.ceil(abs(t1-t2)/48)
    # 如果是closeness, delta_d = 0， 如果是 period， delta_d = abs(day2 - day1)
    delta_d = 0
    if period:
        delta_d = abs(df.loc[t2-1, 'day'] - df.loc[t1-1, 'day'])
    lambda_1 = pow(rho_1, delta_h) * pow(rho_2, delta_d)
    return lambda_1


# 计算天气的相似性 t: 1-1152
def lambda_weather(t1, t2):
    global df, max_temperature, min_temperature, max_pm25, min_pm25
    # 是用时间片的 weather 还是当天的 weather
    weather_t1 = df.loc[t1-1, 'weather']
    weather_t2 = df.loc[t2-1, 'weather']
    # alpha_1……6
    x = 10
    if weather_t1 == weather_t2:
        lambda_2 = 1
    elif weather_t1 == 4:
        lambda_2 = weather_t2 / x
    elif weather_t2 == 4:
        lambda_2 = weather_t1 / x
    else:
        lambda_2 = 1 - abs(weather_t2 - weather_t1) / 4

    lambda_3 = 1 - (abs(df.loc[t1-1, 'temperature'] - df.loc[t2-1, 'temperature']) / (max_temperature - min_temperature))
    lambda_4 = 1 - (abs(df.loc[t1-1, 'pm2.5'] - df.loc[t2-1, 'pm2.5']) / (max_pm25 - min_pm25))

    return lambda_2*lambda_3*lambda_4

if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df, flow_total, proportion_list, df_test = get_all_data()
    max_temperature = df['temperature'].max()
    min_temperature = df['temperature'].min()
    max_pm25 = df['pm2.5'].max()
    min_pm25 = df['pm2.5'].min()
    predict_batch([833, 834, 835, 836])

    # t1 = 2
    # t2 = 50
    # print(lambda_time(t1, t2))
    # print(lambda_weather(t1, t2))

    # predict(1060, 8)
    # train, test = [], []
    # x_train, y_train = train.iloc[:, 0:-1], train.loc[:, 'count']
    # x_test, y_test = test.iloc[:, 0:-1], test.loc[:, 'count']

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

