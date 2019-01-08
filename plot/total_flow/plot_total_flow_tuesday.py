# 每个周二的城市总订单量随时间变化图

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
from pylab import *
from datetime import datetime


# 获取每个周二 一周数据的城市总流量, 画出城市总流量随时间片变化的曲线
def plot_total_flow_change_tuesday():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df_1 = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 2, 23)]
    df_2 = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 1)]
    df_3 = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 8)]
    df_4 = df_set.loc[pd.to_datetime(df_set['date']) == datetime(2016, 3, 15)]

    x = range(1, 49)
    y1 = [x/1000 for x in list(df_1['count'].values)]
    y2 = [x/1000 for x in list(df_2['count'].values)]
    y3 = [x/1000 for x in list(df_3['count'].values)]
    y4 = [x/1000 for x in list(df_4['count'].values)]

    # plt.title('订单量随时间片变化曲线', fontproperties=font)
    plt.figure(figsize=(9, 6))
    plt.xlabel('时间片')
    plt.ylabel('订单量x$10^{3}$')
    plt.plot(x, y1, linestyle='-', linewidth=2, label=u'2016.2.23周二')
    plt.plot(x, y2, linestyle='-', linewidth=2, label=u'2016.3.1周二')
    plt.plot(x, y3, linestyle='-', linewidth=2, label=u'2016.3.8周二')
    plt.plot(x, y4, linestyle='-', linewidth=2, label=u'2016.3.15周二')

    def formatnum(x, pos):
        return '$%.1f$x$10^{3}$' % (x / 1000)

    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_total_flow_change_tuesday()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))