# 画出城市总订单量在24天每个weekday的订单量

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


# 获取24天的城市总流量, 画出每个weekday的平均城市总流量随时间片变化的曲线
def plot_total_flow_avg_weekday():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df_set['week_day'] = pd.to_datetime(df_set['date']).map(lambda x: x.isoweekday())
    df_monday = df_set.loc[df_set['week_day'] == 1].groupby('half_hour').mean()
    df_tuesday = df_set.loc[df_set['week_day'] == 2].groupby('half_hour').mean()
    df_wednesday = df_set.loc[df_set['week_day'] == 3].groupby('half_hour').mean()
    df_thursday = df_set.loc[df_set['week_day'] == 4].groupby('half_hour').mean()
    df_friday = df_set.loc[df_set['week_day'] == 5].groupby('half_hour').mean()
    df_saturday = df_set.loc[df_set['week_day'] == 6].groupby('half_hour').mean()
    df_sunday = df_set.loc[df_set['week_day'] == 7].groupby('half_hour').mean()

    x = range(1, 49)
    y1 = [x/1000 for x in list(df_monday['count'].values)]
    y2 = [x/1000 for x in list(df_tuesday['count'].values)]
    y3 = [x/1000 for x in list(df_wednesday['count'].values)]
    y4 = [x/1000 for x in list(df_thursday['count'].values)]
    y5 = [x/1000 for x in list(df_friday['count'].values)]
    y6 = [x/1000 for x in list(df_saturday['count'].values)]
    y7 = [x/1000 for x in list(df_sunday['count'].values)]

    # f, ax = plt.subplots(1, 1)
    # plt.title('订单量随时间片变化曲线', fontproperties=font)
    plt.xlabel('时间片')
    plt.ylabel('订单量x$10^{3}$')
    plt.plot(x, y1, linestyle='-', linewidth=2, label=u'周一')
    plt.plot(x, y2, linestyle='-', linewidth=2, label=u'周二')
    plt.plot(x, y3, linestyle='-', linewidth=2, label=u'周三')
    plt.plot(x, y4, linestyle='-', linewidth=2, label=u'周四')
    plt.plot(x, y5, linestyle='-', linewidth=2, label=u'周五')
    plt.plot(x, y6, linestyle='-', linewidth=2, label=u'周六')
    plt.plot(x, y7, linestyle='-', linewidth=2, label=u'周日')
    # plt.plot(x, list(df_saturday['count'].values), marker='*', linestyle='-', linewidth=2, label=u'周六')
    # plt.plot(x, list(df_sunday['count'].values), marker='d', linestyle='-', linewidth=2, label=u'周日')

    # def formatnum(x, pos):
    #     return '$%.1f$x$10^{3}$' % (x / 1000)

    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_total_flow_avg_weekday()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))







