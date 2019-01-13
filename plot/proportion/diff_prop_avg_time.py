# 取2.29-3.4五天的数据，按照时间片对比例矩阵取平均，画出某一时间片同其之前时间片的相似度变化曲线

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import extract_feature.gen_proportion as gp
import numpy as np
import pandas as pd
from datetime import datetime


# 两个比例的距离越大，说明比例矩阵越不相似
def distance(prop1, prop2):
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    d = np.sqrt(np.sum(np.square(prop1 - prop2)))
    return d


# 先得到每个时间片的平均比例矩阵（工作日），看时间片t和之前时间片的矩阵距离变化情况
def plot_prop_difference_avg_time_in_time_t(t, pre_len):
    global proportion_list
    day_list = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 16, 17, 18, 21, 22, 23, 24]
    prop_list = []
    for i in range(48):
        # i=16
        prop_sum = np.zeros((58, 58))
        for j in range(len(day_list)):
            prop_time = proportion_list[48*(day_list[j]-1)+i]
            prop_sum += prop_time
        prop_avg = prop_sum/48
        prop_list.append(prop_avg)
    print(len(prop_list))
    prop_t = prop_list[t]
    dist = []
    for i in range(t-1, t-pre_len-1, -1):
        # i = (i + 48) % 48     # 有没有都一样
        dist.append(distance(prop_t, prop_list[i]))
    x = np.arange(1, pre_len+1)
    y = np.array(dist)
    # plt.title('时间片t的比例矩阵与之前pre_len个的距离')
    plt.xlabel('前面时间片')
    plt.xlim((0, len(x)+1))
    plt.ylabel('平均比例矩阵相似度')
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)

    plt.legend(loc='best')
    plt.show()


def plot_prop_difference_avg_time(pre_len):
    global proportion_list
    day_list = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 16, 17, 18, 21, 22, 23, 24]
    prop_list = []
    for i in range(48):
        prop_sum = np.zeros((58, 58))
        for j in range(len(day_list)):
            prop_time = proportion_list[48*(day_list[j]-1)+i]
            prop_sum += prop_time
        prop_avg = prop_sum/48
        prop_list.append(prop_avg)
    print(len(prop_list))

    dist_sum = np.zeros((1, pre_len))
    for ti in range(48):
        prop_t = prop_list[ti]
        dist = []
        for i in range(ti - 1, ti - pre_len - 1, -1):
            i = (i+48) % 48
            dist.append(distance(prop_t, prop_list[i]))
        dist_sum += np.array(dist)
    dist_avg = dist_sum/48

    x = np.arange(1, pre_len+1)
    y = dist_avg[0]

    plt.title('时间片t的比例矩阵与之前pre_len个的距离')
    plt.xlabel('前面时间片')
    plt.xlim((0, len(x)+1))
    plt.ylabel('平均比例矩阵距离')
    plt.plot(x, y, color='red', marker='o', linestyle='-', linewidth=2)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = gp.get_proportion()
    plot_prop_difference_avg_time_in_time_t(24, 10)
    # plot_prop_difference_avg_time(20)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))







