# 分别从城市和区域的角度画出交通拥堵信息随着day的变化

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


# 从城市的角度，交通拥堵信息在这24天的变化
def plot_total_traffic_change_with_day():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
    del df['time']
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').sum()
    # df = df.groupby(['district_id', 'date']).sum()

    max1 = df['tj_level_1'].max()
    max2 = df['tj_level_2'].max()
    max3 = df['tj_level_3'].max()
    max4 = df['tj_level_4'].max()

    x = range(1, 25)
    y1 = [x/max1 for x in list(df['tj_level_1'].values)]
    y2 = [x/max2 for x in list(df['tj_level_2'].values)]
    y3 = [x/max3 for x in list(df['tj_level_3'].values)]
    y4 = [x/max4 for x in list(df['tj_level_4'].values)]

    fig, axs = plt.subplots(4, 1, sharex=True)
    # fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    plt.xlabel('天')
    plt.ylabel('城市总订单量（x$10^{5}$）')

    # Plot each graph, and manually set the y tick values
    # plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2)
    axs[0].plot(x, y1, linestyle='-', linewidth=2)
    axs[0].set_ylim(0, 1.2)
    axs[0].set_ylabel('tc_level_1')
    axs[1].plot(x, y2, linestyle='-', linewidth=2)
    axs[1].set_ylim(0, 1.2)
    axs[1].set_ylabel('tc_level_2')
    axs[2].plot(x, y3, linestyle='-', linewidth=2)
    axs[2].set_ylim(0, 1.2)
    axs[2].set_ylabel('tc_level_3')
    axs[3].plot(x, y4, linestyle='-', linewidth=2)
    axs[3].set_ylim(0, 1.2)
    axs[3].set_ylabel('tc_level_4')

    # plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()


# 从区域的角度，交通拥堵信息在这24天的变化，取区域 1, 10, 37, 54
def plot_region_traffic_change_with_day():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
    del df['time']
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['district_id', 'date']).sum()

    regions = [1, 10, 37, 54]

    fig, axs = plt.subplots(4, 1, sharex=True)
    # fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    plt.xlabel('天')
    plt.ylabel('城市总订单量（x$10^{5}$）')
    axs[0].set_ylim(0, 1.2)
    axs[0].set_ylabel('tc_level_1')
    axs[1].set_ylim(0, 1.2)
    axs[1].set_ylabel('tc_level_2')
    axs[2].set_ylim(0, 1.2)
    axs[2].set_ylabel('tc_level_3')
    axs[3].set_ylim(0, 1.2)
    axs[3].set_ylabel('tc_level_4')

    for i in range(len(regions)):
        max1, max2, max3, max4 = df.xs((regions[i])).max().values

        x = range(1, 25)
        y1 = [x/max1 for x in list(df.xs((regions[i]))['tj_level_1'].values)]
        y2 = [x/max2 for x in list(df.xs((regions[i]))['tj_level_2'].values)]
        y3 = [x/max3 for x in list(df.xs((regions[i]))['tj_level_3'].values)]
        y4 = [x/max4 for x in list(df.xs((regions[i]))['tj_level_4'].values)]

        # Plot each graph, and manually set the y tick values
        # plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2)

        axs[0].plot(x, y1, linestyle='-', linewidth=2)
        axs[1].plot(x, y2, linestyle='-', linewidth=2)
        axs[2].plot(x, y3, linestyle='-', linewidth=2)
        axs[3].plot(x, y4, linestyle='-', linewidth=2)

    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()



# 从区域的角度，交通拥堵信息在时间片t(t=35)的变化，取区域 8
def plot_region_traffic_change_with_day_of_time_t(t):
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
    df = df.loc[df['time'] == 35]
    df['date'] = pd.to_datetime(df['date'])
    del df['time']
    df = df.groupby(['district_id', 'date']).sum()
    print(df)

    regions = [8]

    fig, axs = plt.subplots(2, 1, sharex=True)
    # fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    plt.xlabel('天')
    plt.ylabel('区域总订单量（x$10^{2}$）')

    axs[0].set_xlim(0, 25)
    axs[0].set_ylabel('tc_level_3')
    # axs[1].set_xlim(0, 25)
    axs[1].set_ylabel('tc_level_4')

    for i in range(len(regions)):
        x = range(1, 25)
        y3 = [x for x in list(df.xs((regions[i]))['tj_level_3'].values)]
        y4 = [x for x in list(df.xs((regions[i]))['tj_level_4'].values)]

        # Plot each graph, and manually set the y tick values
        # plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2)
        # axs[0].set_ylim(1.5, 4)
        # axs[1].set_ylim(0.8, 2.2)

        axs[0].plot(x, y3, marker='o', linestyle='-', linewidth=2)
        axs[1].plot(x, y4, marker='o', linestyle='-', linewidth=2)
        axs[0].scatter(x[3], y3[3], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        axs[0].scatter(x[10], y3[10], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        axs[0].scatter(x[17], y3[17], color='', marker='o', linestyle='-', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        axs[1].scatter(x[3], y4[3], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        axs[1].scatter(x[10], y4[10], color='', marker='o', linestyle='--', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        axs[1].scatter(x[17], y4[17], color='', marker='o', linestyle='-', linewidth=2, edgecolors='r', s=200)  # 把 corlor 设置为空，通过edgecolors来控制颜色

    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # plot_total_traffic_change_with_day()
    # plot_region_traffic_change_with_day()
    plot_region_traffic_change_with_day_of_time_t(35)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

