# 使用 HA(历史平均) 模型预测所有时间片上城市总的订单量（区分工作日和周末）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
# MAE 平均绝对误差
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from datetime import datetime
import pickle


def gen_total_set():
    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    del df_time['date']
    del df_time['time']
    df = pd.concat([df_weather, df_time], axis=1)
    return df


# 计算历史平均2.23-3.10
def model_ha():
    global x_train, y_train, x_test, y_test, df_data_set

    df_mean = pd.DataFrame()
    cut_point = int(len(x)/24*17)
    df_mean['date'] = df_data_set.loc[cut_point:len(x), 'date']
    df_mean['time'] = df_data_set.loc[cut_point:len(x), 'time']
    df_mean['is_weekend'] = df_data_set.loc[cut_point:len(x), 'is_weekend']
    df_mean['mean_his_day'] = df_data_set.loc[cut_point:len(x), 'mean_his_day']
    df_mean['mean_his_week'] = df_data_set.loc[cut_point:len(x), 'mean_his_week']

    df_mean.set_index(keys=['date', 'time'], inplace=True)
    df_mean.to_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_total.csv')

    mse = mean_squared_error(y_test, df_mean['mean_his_day'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    MAE = mean_absolute_error(y_test, df_mean['mean_his_day'])
    print("MAE: %.4f" % MAE)  # 输出平均绝对误差
    msle = mean_squared_log_error(y_test, df_mean['mean_his_day'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, df_mean['mean_his_day'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    mse = mean_squared_error(y_test, df_mean['mean_his_week'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    MAE = mean_absolute_error(y_test, df_mean['mean_his_week'])
    print("MAE: %.4f" % MAE)  # 输出平均绝对误差
    msle = mean_squared_log_error(y_test, df_mean['mean_his_week'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, df_mean['mean_his_week'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好




# # 计算历史平均2.23-3.10
# def model_ha():
#     global x_train, y_train, df_data_set, x
#     x_train = x_train[['time', 'is_weekend']]
#     train = pd.concat([x_train, y_train], axis=1)
#     mean_total = train.groupby(['is_weekend', 'time']).mean()
#     # mean_total['is_weekend'] = mean_total.index.get_level_values('is_weekend')
#     # mean_total['time'] = mean_total.index.get_level_values('time')
#
#     df_mean = pd.DataFrame()
#     cut_point = int(len(x)/24*17)
#     df_mean['date'] = df_data_set.loc[cut_point:len(x), 'date']
#     df_mean['time'] = df_data_set.loc[cut_point:len(x), 'time']
#     df_mean['is_weekend'] = df_data_set.loc[cut_point:len(x), 'is_weekend']
#     for i in range(cut_point, len(x)):
#         is_weekend = df_mean.loc[i, 'is_weekend']
#         time = df_mean.loc[i, 'time']
#         df_mean.loc[i, 'mean_count'] = mean_total.xs((is_weekend, time))['count']
#
#     df_mean.set_index(keys=['date', 'time'], inplace=True)
#     df_mean.to_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_total.csv')
#
#     mse = mean_squared_error(y_test, df_mean['mean_count'])
#     print("MSE: %.4f" % mse)  # 输出均方误差
#     r2 = r2_score(y_test, df_mean['mean_count'])
#     print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_data_set = gen_total_set()
    x, y = df_data_set.iloc[:, 1:-1], df_data_set.loc[:, 'count'].values
    x_train, y_train = x.iloc[0:int(len(x)/24*17), :], y[0:int(len(y)/24*17)]
    x_test, y_test = x.iloc[int(len(x)/24*17):len(x), :], y[int(len(y)/24*17):len(y)]
    model_ha()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 根据周末还是工作日
# MSE: 2571937.7133
# MAE: 1174.7786
# MSLE: 0.0462
# r^2 on test data : 0.878773

# 每周的这一天
# MSE: 2341561.2597
# MAE: 1097.6146
# MSLE: 0.0450
# r^2 on test data : 0.889632



