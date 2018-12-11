# 统计所有司机在2.23--3.7期间所有的订单量并写入csv文件（driver_id, orderCount）, 数据是 order_data 文件夹
# 将不同订单数量对应的司机数量表示成条形分布图

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Tools import funcTools as ft

dirPath = 'E:\\data\\DiDiData\\data_csv\\order_data\\'
fileList = ft.listdir_nohidden(dirPath)
fileList.sort()
dict_driver = {}
print(fileList)

dict_numCount = {}   # 记录订单数量相等的司机数量

# 取前两周数据进行统计并写入文件
# for i in range(len(fileList)):
for i in range(1):
    fileDate = fileList[i].split('_')[2]
    print(fileDate)
    filePath = os.path.join('%s%s' % (dirPath, fileList[i]))
    print(filePath)
    df = pd.read_csv(filePath)
    for j in range(len(df)):
        if df.driver_id[j] in dict_driver.keys():
            dict_driver[df.driver_id[j]] += 1   # 已有该driver, 订单量 +1
        else:
            dict_driver[df.driver_id[j]] = 1    # 没有该 driver, 添加,订单量=1

    # 计算订单数量相等的司机数量
    for value in dict_driver.values():
        key = value
        if key in dict_numCount.keys():
            dict_numCount[key] += 1
        else:
            dict_numCount[key] = 1

    print(dict_numCount)




    # 一天中,不同订单量的司机数量条形分布图

    n_groups = len(dict_numCount)
    y_values = dict_numCount.values()
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    print(type(index), index)
    bar_width = 0.35
    rects = plt.bar(index, y_values, bar_width, alpha=0.4, color='b', label='driverCount')
    # rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, col or = 'r', label = 'Women')
    plt.xlabel('orderCount')
    plt.ylabel('driverCount')
    plt.title('the number of drivers in each orderCount')
    plt.xticks(index + bar_width, dict_numCount.keys())
    # plt.ylim(0, 40)
    plt.legend()
    plt.tight_layout()
    plt.show()




    df_out = pd.DataFrame({'driver_id': list(dict_driver.keys()), 'order_count': list(dict_driver.values())}).sort_values(by='order_count', axis=0, ascending=True)
    # df_out.sort_values(by='order_count', axis=0, ascending=True)
    df_out.set_index(keys='driver_id', inplace=True)

    print(len(df_out))
    print(df_out.head())
    desPath = 'E:\\data\\DiDiData\\data_csv\\data_csv\\driver_order_count\\'+fileDate
    # df_out.to_csv(desPath)

