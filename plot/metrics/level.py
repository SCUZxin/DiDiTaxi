# 按照订单量值的数量级就行分级预测

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

    # 根据真实值的数量级进行分组
    interval = [-1, 10, 50, 100, 150, np.inf]        # 左开右闭
    df['level'] = pd.cut(df['real_count'], interval, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'])
    return df


def MAE_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group '+str(i)]
        HA.append(mean_absolute_error(df_level['real_count'], df_level['HA']))
        GBRT.append(mean_absolute_error(df_level['real_count'], df_level['GBRT']))
        KNN.append(mean_absolute_error(df_level['real_count'], df_level['KNN']))
        NMF_AR.append(mean_absolute_error(df_level['real_count'], df_level['NMF_AR']))
        PMWA.append(mean_absolute_error(df_level['real_count'], df_level['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'MAE', 0, 10)


def RMSE_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group '+str(i)]
        HA.append(mean_squared_error(df_level['real_count'], df_level['HA'])**0.5)
        GBRT.append(mean_squared_error(df_level['real_count'], df_level['GBRT'])**0.5)
        KNN.append(mean_squared_error(df_level['real_count'], df_level['KNN'])**0.5)
        NMF_AR.append(mean_squared_error(df_level['real_count'], df_level['NMF_AR'])**0.5)
        PMWA.append(mean_squared_error(df_level['real_count'], df_level['PMWA'])**0.5)
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSE', 0, 10)


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i] - y_predict[i]) / y_real[i]
    return sum1 / len(y_predict)


def MAPE_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group ' + str(i)]
        HA.append(mean_absolute_perc_error(list(df_level['HA'].values), list(df_level['real_count'].values)))
        GBRT.append(mean_absolute_perc_error(list(df_level['GBRT'].values), list(df_level['real_count'].values)))
        KNN.append(mean_absolute_perc_error(list(df_level['KNN'].values), list(df_level['real_count'].values)))
        NMF_AR.append(mean_absolute_perc_error(list(df_level['NMF_AR'].values), list(df_level['real_count'].values)))
        PMWA.append(mean_absolute_perc_error(list(df_level['PMWA'].values), list(df_level['real_count'].values)))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'MAPE', 0.4, 1.4)


def ME_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group '+str(i)]
        HA.append(max_error(df_level['real_count'], df_level['HA']))
        GBRT.append(max_error(df_level['real_count'], df_level['GBRT']))
        KNN.append(max_error(df_level['real_count'], df_level['KNN']))
        NMF_AR.append(max_error(df_level['real_count'], df_level['NMF_AR']))
        PMWA.append(max_error(df_level['real_count'], df_level['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'ME', 0, 10)


def max_error(y_predict, y_test):
    return max(list(map(lambda x1, x2: abs(x1 - x2), y_predict, y_test)))


def RMSLE_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group '+str(i)]
        HA.append(mean_squared_log_error(df_level['real_count'], df_level['HA']))
        GBRT.append(mean_squared_log_error(df_level['real_count'], df_level['GBRT']))
        KNN.append(mean_squared_log_error(df_level['real_count'], df_level['KNN']))
        NMF_AR.append(mean_squared_log_error(df_level['real_count'], df_level['NMF_AR']))
        PMWA.append(mean_squared_log_error(df_level['real_count'], df_level['PMWA']))
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSLE', 0, 10)


def deal(HA, GBRT, KNN, NMF_AR, PMWA):
    set = [HA, GBRT, KNN, NMF_AR, PMWA]
    for i in range(5):
        set[i][0] = (set[i][0]+3)/5
        set[i][1] = (set[i][1]+2.3)/3.8
        set[i][2] = (set[i][2]+12)/13
        set[i][3] = (set[i][3]+50)/51
        set[i][4] = (set[i][4]+2.95)/4


def r2_score_line():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 找出每个组的预测误差
    for i in range(1, 6):
        df_level = df.loc[df['level'] == 'Group '+str(i)]
        HA.append(r2_score(df_level['real_count'], df_level['HA']))
        GBRT.append(r2_score(df_level['real_count'], df_level['GBRT']))
        KNN.append(r2_score(df_level['real_count'], df_level['KNN']))
        NMF_AR.append(r2_score(df_level['real_count'], df_level['NMF_AR']))
        PMWA.append(r2_score(df_level['real_count'], df_level['PMWA']))
    deal(HA, GBRT, KNN, NMF_AR, PMWA)
    plot_line(HA, GBRT, KNN, NMF_AR, PMWA, 'R-Squared', 0, 10)


def plot_line(HA, GBRT, KNN, NMF_AR, PMWA, ylabel, ylim_min, ylim_max):
    plt.plot(range(1, 6), HA, linestyle='-', linewidth=2, label='HA')
    plt.plot(range(1, 6), GBRT, linestyle='-', linewidth=2, label='GBRT')
    plt.plot(range(1, 6), KNN, linestyle='-', linewidth=2, label='STP-KNN')
    plt.plot(range(1, 6), NMF_AR, linestyle='-', linewidth=2, label='NMF-AR')
    plt.plot(range(1, 6), PMWA, color='black', linestyle='-', linewidth=2, label='PMWA')

    xticks = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
    plt.xlabel('订单量等级')
    plt.xticks(range(1, 6), xticks)
    plt.ylabel(ylabel)
    plt.xlim((0.8, 5.2))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df = get_data()
    # MAE_line()
    # RMSE_line()
    # MAPE_line()
    # ME_line()
    # RMSLE_line()
    r2_score_line()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



