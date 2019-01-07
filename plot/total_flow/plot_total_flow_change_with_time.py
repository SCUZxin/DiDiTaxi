# 某一天的城市总流量随时间片变化情况
# 24天的城市平均总流量随时间片变化情况


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
from pylab import *

from datetime import datetime
import pickle


# 获取3.9日城市总流量, 画出城市总流量随时间片变化的曲线
def plot_total_flow_change_3_9():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df_set = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 9)]

    x = range(1, 49)
    y = list(df_set['count'].values)

    f, ax = plt.subplots(1, 1)
    # plt.title('订单量随时间片变化曲线', fontproperties=font)
    plt.xlabel('时间片', fontproperties=font)
    plt.ylabel('订单量', fontproperties=font)
    plt.plot(x, y, color='b', marker='o', linestyle='-', linewidth=2)

    def formatnum(x, pos):
        return '$%.1f$x$10^{3}$' % (x / 1000)

    formatter = FuncFormatter(formatnum)
    ax.yaxis.set_major_formatter(formatter)

    plt.legend(loc='best')
    plt.show()


# 获取2.29-3.6一周数据的城市总流量, 画出城市总流量随时间片变化的曲线
def plot_total_flow_change_a_week():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df_monday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 2, 29)]
    df_tuesday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 1)]
    df_wednesday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 2)]
    df_thursday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 3)]
    df_friday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 4)]
    df_saturday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 5)]
    df_sunday = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 6)]

    x = range(1, 49)

    f, ax = plt.subplots(1, 1)
    # plt.title('订单量随时间片变化曲线', fontproperties=font)
    plt.xlabel('时间片', fontproperties=font)
    plt.ylabel('订单量', fontproperties=font)
    plt.plot(x, list(df_monday['count'].values), marker='o', linestyle='-', linewidth=2, label=u'周一')
    plt.plot(x, list(df_tuesday['count'].values), marker='v', linestyle='-', linewidth=2, label=u'周二')
    plt.plot(x, list(df_wednesday['count'].values), marker='x', linestyle='-', linewidth=2, label=u'周三')
    plt.plot(x, list(df_thursday['count'].values), marker='s', linestyle='-', linewidth=2, label=u'周四')
    plt.plot(x, list(df_friday['count'].values), marker='+', linestyle='-', linewidth=2, label=u'周五')
    plt.plot(x, list(df_saturday['count'].values), marker='*', linestyle='-', linewidth=2, label=u'周六')
    plt.plot(x, list(df_sunday['count'].values), marker='d', linestyle='-', linewidth=2, label=u'周日')
    # plt.plot(x, list(df_monday['count'].values), color='b', marker='o', linestyle='-', linewidth=2, label=u'周一')
    # plt.plot(x, list(df_tuesday['count'].values), color='#A0522D', marker='v', linestyle='-', linewidth=2, label=u'周二')
    # plt.plot(x, list(df_wednesday['count'].values), color='yellow', marker='x', linestyle='-', linewidth=2, label=u'周三')
    # plt.plot(x, list(df_thursday['count'].values), color='#40E0D0', marker='s', linestyle='-', linewidth=2, label=u'周四')
    # plt.plot(x, list(df_friday['count'].values), color='indigo', marker='+', linestyle='-', linewidth=2, label=u'周五')
    # plt.plot(x, list(df_saturday['count'].values), color='r', marker='*', linestyle='-', linewidth=2, label=u'周六')
    # plt.plot(x, list(df_sunday['count'].values), color='hotpink', marker='d', linestyle='-', linewidth=2, label=u'周日')


    def formatnum(x, pos):
        return '$%.1f$x$10^{3}$' % (x / 1000)

    formatter = FuncFormatter(formatnum)
    ax.yaxis.set_major_formatter(formatter)

    plt.legend(loc='best')
    plt.show()


# 获取所有数据中城市总流量, 画出城市总流量随时间片变化的曲线
def plot_total_flow_change():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df_set = df_set.groupby('half_hour').sum()

    x = range(1, 49)
    y = list(df_set['count'].values)
    print(y)

    # fig = plt.figure(1, figsize=(10, 6))
    # import mpl_toolkits.axisartist.axislines as axislines
    # ax1 = axislines.Subplot(fig, 111)
    # fig.add_subplot(ax1)
    # # 设置刻度
    # ax1.set_xticks([10, 10000, 100000, 350000])
    # ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    f, ax = plt.subplots(1, 1)

    # plt.title('订单量随时间片变化曲线', fontproperties=font)
    plt.xlabel('时间片', fontproperties=font)
    plt.ylabel('订单量', fontproperties=font)
    plt.plot(x, y, color='b', marker='o', linestyle='-', linewidth=2)

    def formatnum(x, pos):
        return '$%.1f$x$10^{4}$' % (x / 10000)
    formatter = FuncFormatter(formatnum)
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_total_flow_change_a_week()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))





