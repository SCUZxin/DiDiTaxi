# 计算出所有 OD 对的总流量（2.23-3.10），按照流量大小进行排序
# 得到的曲线横坐标是流量值，纵坐标是OD对占比，曲线表示流量值小于x的OD对占比多少

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
from pylab import *

from datetime import datetime
import pickle


# 获取每一个OD对的总流量以及总共的OD对的数量
def get_od_flow_sum():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    df_set = df_set.loc[pd.to_datetime(df_set['date']) <= datetime(2016, 3, 10)] \
        .reset_index(drop=True)
    df_set = df_set.groupby(['start_district_id', 'dest_district_id'])['count'].sum()
    df = pd.DataFrame()
    df['start_district_id'] = df_set.index.get_level_values('start_district_id')
    df['dest_district_id'] = df_set.index.get_level_values('dest_district_id')
    df['count'] = list(df_set.values)
    # 得到每一个OD对的总流量，保存在 df 中
    return df


# 画出小于x轴所示流量值的OD对占比的曲线
def plot_flow_proportion():
    global df_sum
    sum_od_count = len(df_sum)  # 总共有这么多个OD对, 共2722对
    flow_ranked = sorted(list(df_sum['count'].values))
    x = np.array(sorted(list(set(flow_ranked))))
    count = []  # 保存每一个值出现的次数，顺序与x中一致, 从小到大, len(count) = 816
    for i in x:
        count.append(flow_ranked.count(i))
    prop_array = np.array(list(map(lambda i: i / sum_od_count, count)))
    y = []
    sum1 = 0
    for i in prop_array:
        sum1 += i
        y.append(sum1)
    y = np.array(y)

    print(len(y))
    get_OD_below_80_perc(x, y)

    # fig = plt.figure(1, figsize=(10, 6))
    # import mpl_toolkits.axisartist.axislines as axislines
    # ax1 = axislines.Subplot(fig, 111)
    # fig.add_subplot(ax1)
    # # 设置刻度
    # ax1.set_xticks([10, 10000, 100000, 350000])
    # ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # 全部的
    # plt.title('OD对流量值低于x的比例', fontproperties=font)
    # plt.xlabel('总的订单量X', fontproperties=font)
    # plt.ylabel('低于该订单量的OD对比例', fontproperties=font)
    # plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='比例变化曲线')
    # plt.scatter(x[780], y[780], color='b', linewidths=5)
    # plt.text(x[780],  y[780], x[780], ha='center', va='bottom', fontsize=20)
    # plt.text(x[800],  y[600], y[780], ha='center', va='bottom', fontsize=20)
    # plt.legend(loc='best')
    # plt.show()

    # 挪动坐标位置
    ax = plt.gca()
    # 去掉边框
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # 移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))



    # 局部放大
    # plt.title('起止对流量比例', fontproperties=font)
    plt.xlabel('订单量', fontproperties=font)
    plt.ylabel('OD对数量比例', fontproperties=font)
    # plt.plot(x[0:700], y[0:700], color='blue', linestyle='-', linewidth=2, label='比例变化曲线')
    plt.plot(x[0:700], y[0:700], color='blue', linestyle='-', linewidth=2)



    plt.scatter(x[305], y[305], color='r', linewidths=4)
    # plt.text(x[305],  y[305], x[305], ha='center', va='bottom', fontsize=20)
    plt.annotate("(%s,%.2f)" % (x[305], y[305]), xy=(x[305], y[305]), xytext=(10, 0), textcoords='offset points')
    # plt.annotate("(%s,%s)" , xy=(x[305], y[305]), xytext=(-20, 10), textcoords='offset points')
    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()

    # 再放大
    # plt.title('OD对流量值低于x的比例', fontproperties=font)
    # plt.xlabel('总的订单量X', fontproperties=font)
    # plt.ylabel('低于该订单量的OD对比例', fontproperties=font)
    # plt.plot(x[0:500], y[0:500], color='red', linestyle='--', linewidth=2, label='比例变化曲线')
    # plt.scatter(x[64], y[64], color='b', linewidths=5)
    # plt.text(x[64],  y[64], x[64], ha='center', va='bottom', fontsize=20)
    # plt.legend(loc='upper left')
    # plt.show()


# 将低于订单量在最低的80%的OD对保存起来
def get_OD_below_80_perc(x, y):
    try:
        od_below_80perc_list = pickle.load(open('E:\\data\\DiDiData\\data_csv\\plot\\od_below_80perc_list.pkl', 'rb'))
    except FileNotFoundError:
        threshold_count = 0
        for i in range(len(x)):
            if y[i] > 0.8:
                threshold_count = x[i]
                break
        global df_sum
        od_below_80perc_list = []
        for i in range(len(df_sum)):
            if df_sum.loc[i, 'count'] <= threshold_count:
                s = df_sum.loc[i, 'start_district_id']
                d = df_sum.loc[i, 'dest_district_id']
                od_below_80perc_list.append(str(s)+'-'+str(d))
        pickle.dump(od_below_80perc_list, open('E:\\data\\DiDiData\\data_csv\\plot\\od_below_80perc_list.pkl', 'wb'))

    # print(od_below_80perc_list)
    # print('threshold_count', threshold_count)
    # print(df_sum)
    # print(len(od_below_80perc_list))      # 共2178 OD 对
    return od_below_80perc_list


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_sum = get_od_flow_sum()
    plot_flow_proportion()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 结论（2.23-3.10 共17天的数据）：
# 约 60% 的OD对的流量值小于 60,  第59.999%的OD对的订单量是65, x[64] = 65
# 约 80% 的OD对的流量值小于 400, 第80.015%的OD对的订单量是418, x[305] = 418


