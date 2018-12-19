# 计算某一时刻，两天的比例矩阵的相似度，采用欧几里得距离
# 确定是时刻：17  （8:30-9:00）
# 分别提取2.29-3.06连续一周的该时刻的比例矩阵

import extract_feature.gen_proportion as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

def measure_difference(t):
    global proportion_list
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    time_list = [x for x in range(48*6+t, 48*13, 48)]
    prop_list = []
    for t in time_list:
        prop_list.append(proportion_list[t])
    difference = []
    print(distance(prop_list[0], prop_list[1]))
    print(distance(prop_list[0], prop_list[6]))
    for i in range(1, len(prop_list)):
        dist = distance(prop_list[0], prop_list[i])
        difference.append(dist)

    print(difference)
    t = np.arange(1, 7)
    y = np.array(difference)
    plt.plot(t, y, color='red', linestyle='--', linewidth=2, label='the difference of day of week(1-2,1-3,1-4,1-5,1-6,1-7)')

    plt.legend(loc='upper left')
    plt.show()


def distance(prop1, prop2):
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    d = np.sqrt(np.sum(np.square(prop1 - prop2)))
    return d


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = gp.get_proportion()
    measure_difference(24)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


