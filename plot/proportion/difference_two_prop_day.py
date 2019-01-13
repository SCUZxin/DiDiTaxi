# 计算某一时刻，两天的比例矩阵的相似度，采用欧几里得距离
# measure_difference_day 是2.29-3.06连续一周的每一天比例矩阵之和的欧氏距离相似度
# 确定是时刻：17  （8:30-9:00）
# 分别提取2.29-3.06连续一周的该时刻的比例矩阵

import extract_feature.gen_proportion as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

from datetime import datetime


# 一周中在时间片t周一分别同2-7的比例矩阵
def measure_difference(t):
    global proportion_list
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
    plt.plot(t, y, color='red', linestyle='--', linewidth=2, label='the difference of day of week in time t(1-2,1-3,1-4,1-5,1-6,1-7)')

    plt.legend(loc='upper left')
    plt.show()


# 把一天所有的比例矩阵相加得到一天的比例矩阵，再进行每天的相似度计算，而不是某一时刻
def measure_difference_day():
    global proportion_list
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    time_list = [x for x in range(48*6+0, 48*13)]
    print(len(time_list))
    prop_list = [[0]*59]*59
    prop_sum_list = []
    for i in range(len(time_list) + 1):
        if i % 48 == 0 and i != 0:
            prop_sum_list.append(prop_list)
            prop_list = [[0] * 59] * 59
        if i == len(time_list):
            break
        prop_list = list(map(lambda x: list(map(lambda x1:x1[0]+x1[1], zip(x[0], x[1]))), zip(prop_list, proportion_list[time_list[i]])))
    difference = []
    print(distance(prop_sum_list[0], prop_sum_list[1]))
    print(distance(prop_sum_list[0], prop_sum_list[6]))
    for i in range(1, len(prop_sum_list)):
        dist = distance(prop_sum_list[0], prop_sum_list[i])
        difference.append(dist)

    print(difference)
    t = np.arange(1, 7)
    y = np.array(difference)
    plt.plot(t, y, color='red', linestyle='--', linewidth=2, label='the difference of day of week(1-2,1-3,1-4,1-5,1-6,1-7)')

    plt.legend(loc='upper left')
    plt.show()


# 取所有工作日，除3.8日，最后一天与前面天的对应时间片矩阵距离取平均除以48
def measure_difference_day_avg():
    global proportion_list
    day_list = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 16, 17, 18, 21, 22, 23, 24]
    # day_list = [7, 8, 9, 10, 11]
    day_list = [3, 10, 17, 24]
    prop_list = []
    for i in range(len(day_list)):
        prop_day = proportion_list[48*(day_list[i]-1):48*(day_list[i]-1)+48]
        prop_list.append(prop_day)
    prop_first = prop_list[len(day_list)-1]
    dist_avg = []
    for i in range(len(prop_list)-2, -1, -1):
        prop_comp_day = prop_list[i]
        dist_sum = 0
        for j in range(48):
            dist_sum += distance(prop_first[j], prop_comp_day[j])
        dist_avg.append(dist_sum/48)
    x = np.arange(1, len(prop_list))
    y = np.array(dist_avg)
    print(y)
    plt.title('最后一天与之前的平均矩阵距离')
    plt.xlabel('天')
    plt.xlim((0, len(day_list)))
    plt.ylabel('平均比例矩阵距离')
    plt.plot(x, y, color='red', marker='o', linestyle='-', linewidth=2)

    plt.legend(loc='best')
    plt.show()



def distance(prop1, prop2):
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    d = np.sqrt(np.sum(np.square(prop1 - prop2)))
    return d


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = gp.get_proportion()
    # measure_difference_day()
    # measure_difference(24)
    measure_difference_day_avg()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


