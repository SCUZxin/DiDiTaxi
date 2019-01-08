# PM2.5对城市总订单量的影响，某几个时间片上，订单量随每一天的变化
# 35,36,37时间片，3.6日是重度污染，2.28是良

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
from pylab import *
from datetime import datetime


# 在时间片35,36,37，获取城市总订单量每一天的变化
def plot_total_flow_change_with_day_of_time():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv')
    print(df.head())
    time_slice = [35, 36, 37]
    plt.figure(figsize=(12, 6))
    x = range(1, 25)
    for i in range(len(time_slice)):
        df1 = df.loc[df['half_hour'] == time_slice[i]]
        y = [x/1000 for x in df1['count'].values]
        plt.subplot(1, 3, i+1)
        plt.plot(x, y, linestyle='-', marker='o', linewidth=2)
        plt.scatter(x[5], y[5], color='', marker='o', linewidth=2, edgecolors='r', s=150)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        plt.scatter(x[12], y[12], color='', marker='o', linewidth=2, edgecolors='r', s=150)  # 把 corlor 设置为空，通过edgecolors来控制颜色
        plt.title('时间片'+str(time_slice[i]+1))
        plt.xlim((0, 25))
        plt.xlabel('天')
        plt.ylabel('订单量x$10^{3}$')
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_total_flow_change_with_day_of_time()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


