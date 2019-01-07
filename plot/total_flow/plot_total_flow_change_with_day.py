# 城市总订单量在这24天的变化

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


# 城市总订单量在这24天的变化
def plot_flow_change_with_day():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    df = df.groupby('date').sum()
    y = [x/100000 for x in list(df['count'].values)]
    x = range(1, 25)

    # f, ax = plt.subplots(1, 1)
    plt.xlabel('天')
    plt.ylabel('城市总订单量（x$10^{5}$）')
    # plt.plot(x[0:700], y[0:700], color='red', linestyle='-', linewidth=2, label='比例变化曲线')
    plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2)

    # def formatnum(x, pos):
    #     return '$%.1f$x$10^{4}$' % (x / 10000)
    # formatter = FuncFormatter(formatnum)
    # ax.yaxis.set_major_formatter(formatter)

    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_flow_change_with_day()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


