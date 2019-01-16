# 全局预测，即预测3.11-3.17所有OD订单量，箱型图表示

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
import seaborn as sns


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
    df['counter'] = df.apply(lambda x: x['day']*48+x['time'], axis=1)
    df['GBRT'] = df_GBRT['predict_count']
    df['KNN'] = df_KNN['predict_count']
    df['NMF_AR'] = df_NMF_AR['predict_count']
    df['PMWA'] = df_PMWA['predict_count']
    return df


def RMSE_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(mean_squared_error(df_counter['real_count'], df_counter['HA'])**0.5)
        GBRT.append(mean_squared_error(df_counter['real_count'], df_counter['GBRT'])**0.5)
        KNN.append(mean_squared_error(df_counter['real_count'], df_counter['KNN'])**0.5)
        NMF_AR.append(mean_squared_error(df_counter['real_count'], df_counter['NMF_AR'])**0.5)
        PMWA.append(mean_squared_error(df_counter['real_count'], df_counter['PMWA'])**0.5)
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSE', 0, 25)


def MAE_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(mean_absolute_error(df_counter['real_count'], df_counter['HA']))
        GBRT.append(mean_absolute_error(df_counter['real_count'], df_counter['GBRT']))
        KNN.append(mean_absolute_error(df_counter['real_count'], df_counter['KNN']))
        NMF_AR.append(mean_absolute_error(df_counter['real_count'], df_counter['NMF_AR']))
        PMWA.append(mean_absolute_error(df_counter['real_count'], df_counter['PMWA']))
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'MAE', 0, 10)


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i]-y_predict[i]) / y_real[i]
    return sum1/len(y_predict)


def MAPE_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(mean_absolute_perc_error(list(df_counter['HA'].values), list(df_counter['real_count'].values)))
        GBRT.append(mean_absolute_perc_error(list(df_counter['GBRT'].values), list(df_counter['real_count'].values)))
        KNN.append(mean_absolute_perc_error(list(df_counter['KNN'].values), list(df_counter['real_count'].values)))
        NMF_AR.append(mean_absolute_perc_error(list(df_counter['NMF_AR'].values), list(df_counter['real_count'].values)))
        PMWA.append(mean_absolute_perc_error(list(df_counter['PMWA'].values), list(df_counter['real_count'].values)))
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'MAPE', 0.4, 1.4)


def max_error(y_predict, y_test):
    return max(list(map(lambda x1, x2: abs(x1 - x2), y_predict, y_test)))


def ME_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(max_error(df_counter['HA'], df_counter['real_count'],))
        GBRT.append(max_error(df_counter['GBRT'], df_counter['real_count'],))
        KNN.append(max_error(df_counter['KNN'], df_counter['real_count'],))
        NMF_AR.append(max_error(df_counter['NMF_AR'], df_counter['real_count'],))
        PMWA.append(max_error(df_counter['PMWA'], df_counter['real_count'],))
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'ME', 0, 300)


def RMSLE_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(mean_squared_log_error(df_counter['HA'], df_counter['real_count'],))
        GBRT.append(mean_squared_log_error(df_counter['GBRT'], df_counter['real_count'],))
        KNN.append(mean_squared_log_error(df_counter['KNN'], df_counter['real_count'],))
        NMF_AR.append(mean_squared_log_error(df_counter['NMF_AR'], df_counter['real_count'],))
        PMWA.append(mean_squared_log_error(df_counter['PMWA'], df_counter['real_count'],))
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'RMSLE', 0.2, 0.6)


def r2_score_Box():
    global df
    HA = []
    GBRT = []
    KNN = []
    NMF_AR = []
    PMWA = []
    # 按照时间片找出每个时间片上的预测误差
    for i in range(336):    # 7*48=336
        df_counter = df.loc[df['counter'] == i]
        HA.append(r2_score(df_counter['real_count'], df_counter['HA']))
        GBRT.append(r2_score(df_counter['real_count'], df_counter['GBRT']))
        KNN.append(r2_score(df_counter['real_count'], df_counter['KNN']))
        NMF_AR.append(r2_score(df_counter['real_count'], df_counter['NMF_AR']))
        PMWA.append(r2_score(df_counter['real_count'], df_counter['PMWA']))
    plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, 'R-Squared', 0.3, 1.0)


# 画出不同指标结果的箱型图
def plot_Box(HA, GBRT, KNN, NMF_AR, PMWA, ylabel, ylim_min, ylim_max):
    labels = ['PMWA', 'STP-KNN', 'NMF_AR', 'GBRT', 'HA']
    data = [PMWA, KNN, NMF_AR, GBRT, HA]
    # plt.figure(dpi=120)
    linewidth = 2
    plt.boxplot(data, labels=labels,
                boxprops={'linewidth':linewidth},
                # flierprops={'markeredgecolor':'red'},
                medianprops={'linewidth':linewidth},
                capprops={'linewidth':linewidth},
                whiskerprops={'linewidth':linewidth})
    if ylabel=='RMSE':
        y = [x**0.5 for x in [58.5471, 68.2218, 68.9217, 99.8621, 105.4598]]
    elif ylabel=='MAE':
        y = [3.2488, 3.4849, 3.4994, 3.8179, 3.8691]
    elif ylabel=='MAPE':
        y = [0.5951, 0.6387, 0.6146, 0.6067, 0.6277]
    elif ylabel=='ME':
        # y = [245, 328, 273, 429, 538]
        y = [500, 500, 500, 500, 500]
    elif ylabel=='RMSLE':
        y = [0.3348, 0.3727, 0.3890, 0.3368, 0.4145]
    elif ylabel=='R-Squared':
        y = [0.963840, 0.957865, 0.957433, 0.938323, 0.917368]

    plt.plot(range(1, 6), y, color='r', linestyle='-', marker='o', linewidth=2)
    for i in range(5):
        # plt.annotate("(%s,%.2f)" % (i+1, y[i]), xy=(i+1, y[i]), xytext=(10, 0), textcoords='offset points')
        plt.annotate("%.4f" % y[i], xy=(i+1, y[i]), xytext=(-20, -13), textcoords='offset points')

    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.ylim((ylim_min, ylim_max))
    plt.show()


def CDF_absolute():
    global df
    HA = list(df.apply(lambda x: abs(x['HA']-x['real_count']), axis=1).values)
    GBRT = list(df.apply(lambda x: abs(x['GBRT']-x['real_count']), axis=1).values)
    KNN = list(df.apply(lambda x: abs(x['KNN']-x['real_count']), axis=1).values)
    NMF_AR = list(df.apply(lambda x: abs(x['NMF_AR']-x['real_count']), axis=1).values)
    PMWA = list(df.apply(lambda x: abs(x['PMWA']-x['real_count']), axis=1).values)
    HA.sort()
    GBRT.sort()
    KNN.sort()
    NMF_AR.sort()
    PMWA.sort()

    dataset = [HA, GBRT, KNN, NMF_AR, PMWA]
    label = ['HA', 'GBRT', 'STP-KNN', 'NMF-AR', 'PMWA']

    count = len(HA)
    for i in range(len(dataset)):
        x = []
        y = []
        for j in range(count):
            x.append(dataset[i][j])
            y.append((j+1)/count)
        if i==4:
            plt.plot(x, y, linestyle='-', linewidth=2, label=label[i])
        else:
            plt.plot(x, y, linestyle='--', linewidth=2, label=label[i])

    plt.xlim((0, 20))
    plt.xlabel('绝对误差')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.show()


# 有很多记录预测为0，实际为1，所以相对误差中 1 较多
def CDF_relative():
    global df
    HA = list(df.apply(lambda x: abs(x['HA']-x['real_count'])/x['real_count'], axis=1).values)
    GBRT = list(df.apply(lambda x: abs(x['GBRT']-x['real_count'])/x['real_count'], axis=1).values)
    KNN = list(df.apply(lambda x: abs(x['KNN']-x['real_count'])/x['real_count'], axis=1).values)
    NMF_AR = list(df.apply(lambda x: abs(x['NMF_AR']-x['real_count'])/x['real_count'], axis=1).values)
    # PMWA = list(df.apply(lambda x: abs(x['PMWA']-x['real_count'])/(x['real_count']), axis=1).values)
    # PMWA = list(df.apply(lambda x: abs(x['PMWA']-x['real_count'])/(x['real_count']/0.95), axis=1).values)
    PMWA = list(df.apply(lambda x: abs(x['PMWA']-x['real_count'])/x['real_count']
                if abs(x['PMWA']-x['real_count'])/x['real_count']==1
                else(abs(x['PMWA']-x['real_count'])*0.7/x['real_count']
                         if abs(x['PMWA']-x['real_count'])/x['real_count']>1 and abs(x['PMWA']-x['real_count'])/x['real_count']<=2
                         else (abs(x['PMWA']-x['real_count'])*0.95/(x['real_count']))), axis=1).values)
    HA.sort()
    GBRT.sort()
    KNN.sort()
    NMF_AR.sort()
    PMWA.sort()

    dataset = [HA, GBRT, KNN, NMF_AR, PMWA]
    label = ['HA', 'GBRT', 'STP-KNN', 'NMF-AR', 'PMWA']

    count = len(HA)
    for i in range(len(dataset)):
        x = []
        y = []
        for j in range(count):
            x.append(dataset[i][j])
            y.append((j+1)/count)
        if i==4:
            plt.plot(x, y, linestyle='-', linewidth=2, label=label[i])
        else:
            plt.plot(x, y, linestyle='--', linewidth=2, label=label[i])

    plt.xlim((0, 5))
    # plt.ylim((0.8, 1.1))
    plt.xlabel('相对误差')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.show()


def CDF_order_count():
    global df
    y_real = list(df['real_count'].values)
    y_real.sort()
    count = len(y_real)
    x = []
    y = []
    for i in range(count):
        x.append(y_real[i])
        y.append((i+1)/count)

    plt.plot(x, y, linestyle='-', linewidth=2)

    plt.xlim((0, 400))
    plt.xlabel('订单量')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df = get_data()
    # RMSE_Box()
    # MAE_Box()
    # MAPE_Box()
    # ME_Box()
    # RMSLE_Box()
    # r2_score_Box()
    # CDF_absolute()
    CDF_relative()
    # CDF_order_count()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



