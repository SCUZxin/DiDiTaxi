# 按照date的顺序计算同一时间片连续两天的比例矩阵的距离，并保存到csv文件

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import extract_feature.gen_proportion as gp
import numpy as np
import pandas as pd
from datetime import datetime


# 按日期顺序计算每一个时间片连续两天的比例矩阵的距离变化，并保存到csv文件
def measure_difference_days_of_time_all():
    differences = []
    date_list = pd.date_range('20160223', '20160317')
    print(date_list)

    df = pd.DataFrame(columns=['time', 'pre_date', 'date', 'distance'])
    for t in range(0, 48):
        difference = measure_difference_days_of_time_t(t)
        for i in range(len(difference)):
            df.loc[t*23+i] = [t, date_list[i], date_list[i+1], difference[i]]
    df.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\proportion_distance.csv')


# 按日期顺序计算在时间片t，连续两天的比例矩阵的距离变化
def measure_difference_days_of_time_t(t):
    global proportion_list
    time_list = [x for x in range(t, 48*24, 48)]
    prop_list = []
    for t in time_list:
        prop_list.append(proportion_list[t])
    difference = []
    for i in range(1, len(prop_list)):
        dist = distance(prop_list[i-1], prop_list[i])
        difference.append(dist)

    return difference


# 两个比例的距离越大，说明比例矩阵越不相似
def distance(prop1, prop2):
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    d = np.sqrt(np.sum(np.square(prop1 - prop2)))
    return d


# 按日期顺序计算在时间片t，连续两天的比例矩阵的距离变化
def plot_prop_difference_days_of_time_t(t):
    global proportion_list
    time_list = [x for x in range(t, 48*24, 48)]
    prop_list = []
    for t in time_list:
        prop_list.append(proportion_list[t])
    difference = []
    for i in range(1, len(prop_list)):
        dist = distance(prop_list[i-1], prop_list[i])
        difference.append(dist)

    print(len(difference))
    t = np.arange(1, 24)
    y = np.array(difference)
    plt.xlabel('天')
    plt.ylabel('比例矩阵间距离')
    plt.plot(t, y, marker='o', linestyle='-', linewidth=2)

    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = gp.get_proportion()
    # measure_difference_days_of_time_all()
    plot_prop_difference_days_of_time_t(16)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


