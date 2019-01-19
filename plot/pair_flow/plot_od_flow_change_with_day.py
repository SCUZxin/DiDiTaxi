# OD 对订单量在这24天的变化

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


# OD对订单量在这24天的变化，分别取9-8，8-9，18-38，21-15
def plot_od_change_with_day():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    del df['time']
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['start_district_id', 'dest_district_id', 'date']).sum()

    y1 = [x/1000 for x in list(df.xs((9, 8))['count'].values)]
    y2 = [x/1000 for x in list(df.xs((8, 9))['count'].values)]
    y3 = [x/1000 for x in list(df.xs((18, 38))['count'].values)]
    y4 = [x/1000 for x in list(df.xs((21, 15))['count'].values)]

    x = range(1, 25)

    # f, ax = plt.subplots(1, 1)
    plt.xlabel('天')
    plt.ylabel('OD对订单量（x$10^{3}$）')
    # plt.plot(x[0:700], y[0:700], color='red', linestyle='-', linewidth=2, label='比例变化曲线')
    plt.plot(x, y1, marker='o', linestyle='-', linewidth=2, label='OD对9-8')
    plt.plot(x, y2, marker='o', linestyle='-', linewidth=2, label='OD对8-9')
    plt.plot(x, y3, marker='o', linestyle='-', linewidth=2, label='OD对18-38')
    plt.plot(x, y4, marker='o', linestyle='-', linewidth=2, label='OD对21-15')

    # def formatnum(x, pos):
    #     return '$%.1f$x$10^{4}$' % (x / 10000)
    # formatter = FuncFormatter(formatnum)
    # ax.yaxis.set_major_formatter(formatter)

    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()


# OD对订单量在时间片t（t=35）这24天的变化，取8-21
def plot_od_change_with_day_in_time_t(t):
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    df = df.loc[df['time'] == t]
    df['date'] = pd.to_datetime(df['date'])
    del df['time']
    df = df.groupby(['start_district_id', 'dest_district_id', 'date']).sum()

    x = range(1, 25)
    y = [x for x in list(df.xs((8, 21))['count'].values)]

    plt.xlabel('天')
    plt.ylabel('OD对订单量')
    plt.xlim((0, 25))
    # plt.plot(x[0:700], y[0:700], color='red', linestyle='-', linewidth=2, label='比例变化曲线')
    plt.plot(x, y, linestyle='-', marker='o', linewidth=2)
    plt.scatter(x[3], y[3], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r',
                   s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
    plt.scatter(x[10], y[10], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r',
                   s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
    plt.scatter(x[17], y[17], color='', marker='o', linestyle='-', linewidth=2, edgecolors='r',
                   s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色

    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_od_change_with_day()
    plot_od_change_with_day_in_time_t(35)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


