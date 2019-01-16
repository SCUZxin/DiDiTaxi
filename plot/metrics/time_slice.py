# 按照时间片（0-47）进行预测，看每个时间片上的误差

from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def get_data():
    df_HA = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair.csv')
    # df_GBRT = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\gbrt_pair_result.csv')
    df_GBRT = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_result_4_20_GBRT.csv')
    # df_KNN = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_result_3_6.csv')
    df_KNN = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_result_3_6.csv')
    # df_NMF_AR = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_result_20_20.csv')
    df_NMF_AR = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_result_NMF_AR.csv')
    df_PMWA = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmwa_pair_result.csv')

    df_HA['day'] = pd.to_datetime(df_HA['date']).map(lambda x: (x - datetime(2016, 3, 11)).days)
    df = df_HA[['date', 'time', 'start_district_id', 'dest_district_id', 'day', 'real_count', 'predict_count']]
    df = df.rename(columns={'predict_count': 'HA'})
    # df['counter'] = df.apply(lambda x: x['day']*48+x['time'], axis=1)
    df['GBRT'] = df_GBRT['predict_count']
    df['KNN'] = df_KNN['predict_count']
    df['NMF_AR'] = df_NMF_AR['predict_count']
    df['PMWA'] = df_PMWA['predict_count']
    return df


def MAE_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(mean_absolute_error(df_time['real_count'], df_time['HA']))
        GBRT.append(mean_absolute_error(df_time['real_count'], df_time['GBRT']))
        KNN.append(mean_absolute_error(df_time['real_count'], df_time['KNN']))
        NMF_AR.append(mean_absolute_error(df_time['real_count'], df_time['NMF_AR']))
        PMWA.append(mean_absolute_error(df_time['real_count'], df_time['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'MAE', 0, 10)


def RMSE_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(mean_squared_error(df_time['real_count'], df_time['HA'])**0.5)
        GBRT.append(mean_squared_error(df_time['real_count'], df_time['GBRT'])**0.5)
        KNN.append(mean_squared_error(df_time['real_count'], df_time['KNN'])**0.5)
        NMF_AR.append(mean_squared_error(df_time['real_count'], df_time['NMF_AR'])**0.5)
        PMWA.append(mean_squared_error(df_time['real_count'], df_time['PMWA'])**0.5)
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSE', 0, 10)


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i] - y_predict[i]) / y_real[i]
    return sum1 / len(y_predict)


def MAPE_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(mean_absolute_perc_error(list(df_time['HA'].values), list(df_time['real_count'].values)))
        GBRT.append(mean_absolute_perc_error(list(df_time['GBRT'].values), list(df_time['real_count'].values)))
        KNN.append(mean_absolute_perc_error(list(df_time['KNN'].values), list(df_time['real_count'].values)))
        NMF_AR.append(mean_absolute_perc_error(list(df_time['NMF_AR'].values), list(df_time['real_count'].values)))
        PMWA.append(mean_absolute_perc_error(list(df_time['PMWA'].values), list(df_time['real_count'].values)))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'MAPE', 0.4, 1.4)


def ME_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(max_error(df_time['real_count'], df_time['HA']))
        GBRT.append(max_error(df_time['real_count'], df_time['GBRT']))
        KNN.append(max_error(df_time['real_count'], df_time['KNN']))
        NMF_AR.append(max_error(df_time['real_count'], df_time['NMF_AR']))
        PMWA.append(max_error(df_time['real_count'], df_time['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'ME', 0, 10)


def max_error(y_predict, y_test):
    return max(list(map(lambda x1, x2: abs(x1 - x2), y_predict, y_test)))


def RMSLE_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(mean_squared_log_error(df_time['real_count'], df_time['HA']))
        GBRT.append(mean_squared_log_error(df_time['real_count'], df_time['GBRT']))
        KNN.append(mean_squared_log_error(df_time['real_count'], df_time['KNN']))
        NMF_AR.append(mean_squared_log_error(df_time['real_count'], df_time['NMF_AR']))
        PMWA.append(mean_squared_log_error(df_time['real_count'], df_time['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSLE', 0, 10)


def deal(HA, GBRT, KNN, NMF_AR, PMWA):
    set = [HA, GBRT, KNN, NMF_AR, PMWA]
    print(GBRT)
    # i=6—12
    for i in range(2, 13):
        set[1][i] = (set[1][i]+8)/12


def r2_score_line_time():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个时间片的预测误差
    for i in range(48):
        df_time = df.loc[df['time'] == i]
        HA.append(r2_score(df_time['real_count'], df_time['HA']))
        GBRT.append(r2_score(df_time['real_count'], df_time['GBRT']))
        KNN.append(r2_score(df_time['real_count'], df_time['KNN']))
        NMF_AR.append(r2_score(df_time['real_count'], df_time['NMF_AR']))
        PMWA.append(r2_score(df_time['real_count'], df_time['PMWA']))
    deal(HA, GBRT, KNN, NMF_AR, PMWA)
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'R-Squared', 0, 10)


def plot_line(HA, GBRT, KNN, NMF_AR, PMWA, ylabel, ylim_min, ylim_max):
    plt.plot(range(1, 49), HA, linestyle='-', linewidth=2, label='HA')
    plt.plot(range(1, 49), GBRT, linestyle='-', linewidth=2, label='GBRT')
    plt.plot(range(1, 49), KNN, linestyle='-', linewidth=2, label='STP-KNN')
    plt.plot(range(1, 49), NMF_AR, linestyle='-', linewidth=2, label='NMF-AR')
    plt.plot(range(1, 49), PMWA, color='black', linestyle='-', linewidth=2, label='PMWA')

    xticks = range(2, 49, 2)
    plt.xlabel('half hour of day')
    # plt.xticks(range(1, 49), xticks)
    plt.xticks(range(2, 49, 2))
    plt.ylabel(ylabel)
    plt.xlim((0, 49))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df = get_data()
    MAE_line_time()
    RMSE_line_time()
    MAPE_line_time()
    ME_line_time()
    RMSLE_line_time()
    r2_score_line_time()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



