# 取不同的pre_len，这几个时间片的比例矩阵与当前时间片的比例矩阵的平均欧氏距离，
# 看看变化图，x轴是取len，y轴是平均欧式距离

import extract_feature.gen_proportion as gp
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


def measure_difference(pre_len):
    global proportion_list
    proportion_list = proportion_list[1:672]
    dist_avg_list = []
    for i in range(len(proportion_list)-pre_len):
        target_prop = proportion_list[i+pre_len]
        dist = 0
        for j in range(pre_len):
            dist = dist + distance(proportion_list[i+j], target_prop)
        dist = dist / pre_len
        dist_avg_list.append(dist)
    dist_avg = np.array(dist_avg_list).mean()
    print(dist_avg)
    return dist_avg


def distance(prop1, prop2):
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    d = np.sqrt(np.sum(np.square(prop1 - prop2)))
    return d


def plot_difference_of_different_pre_len():
    pre_len_list = [x for x in range(1, 10)]
    difference = []
    for pre_len in pre_len_list:
        difference.append(measure_difference(pre_len))

    x = np.array(pre_len_list)
    y = np.array(difference)
    plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='the difference of different pre_len')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = gp.get_proportion()
    plot_difference_of_different_pre_len()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))








