# 2016年2月29日至2016年3月4日连续5天
# 画出给定OD对（eg：9-8，8-9，18-38，21-15）的流量随时间变化曲线

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


def plot_od_change_with_time_5days(sd_pair):
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].isin(pd.date_range('2016/2/29', '2016/3/4'))]
    # 筛选OD对
    s, d = int(sd_pair.split('-')[0]), int(sd_pair.split('-')[1])
    df = df[(df['start_district_id'] == s) & (df['dest_district_id'] == d)].reset_index(drop=True)
    date_list = pd.date_range('2016/2/29', '2016/3/4')
    # 找出值画图
    values = []
    for i in range(len(date_list)):
        value = []
        df_date = df[df['date']==date_list[i]].reset_index(drop=True)
        for i in range(48):
            df_temp = df_date[df_date['time'] == i]['count']
            if len(df_temp) == 1:
                value.append(df_temp.values[0])
            else:
                value.append(0)
        # values.append(value)
        values.extend(value)

    x = range(1, 48*5+1)
    y = values
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, linestyle='-', linewidth=2)
    for i in range(1, 5):
        plt.axvline(x=48*i, color='#d46061', linewidth=1)
    plt.xlim((-2, 48*5+2))
    xticks = [x for x in range(12, 49, 12)]*5
    plt.xticks(range(12, 48*5+1, 12), xticks)
    plt.xlabel('hour_hour of day')
    # plt.ylabel('OD对订单量（x$10^{3}$）')
    plt.ylabel('OD对订单量')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    od_pair = ['9-8', '8-9', '18-38', '21-15']
    for i in range(len(od_pair)):
        plot_od_change_with_time_5days(od_pair[i])
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))




