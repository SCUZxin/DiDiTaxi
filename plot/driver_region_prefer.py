# 画图证明司机接单具有区域属性
# 词云：字代表接单区域（起点）数目，大小代表司机数

from datetime import datetime
import numpy as np
import pandas as pd
import Tools.funcTools as ft
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


# 统计24天每个司机接单区域及其区域数
def sum_driver_region(fileList):
    try:
        df_all = pd.read_csv('E:\\data\\DiDiData\\data_csv\\driver_region\\drivers_all_order.csv')
    except FileNotFoundError:
        for i in range(len(fileList)):
            print(fileList[i])
            filePath = os.path.join(dirPath, fileList[i])
            df = pd.read_csv(filePath)
            df = df[df.dest_district_id != -1]  # 去除目的地为-1的行数据
            df = df[['driver_id', 'start_district_id']]
            if i == 0:
                df_all = df
            else:
                df_all = df_all.append(df)
        df_all.to_csv('E:\\data\\DiDiData\\data_csv\\driver_region\\drivers_all_order.csv')
        print(len(df_all))

    df_all = df_all.dropna(axis=0)   # 排除不正确的司机id
    drivers = df_all['driver_id'].unique()
    driver_start_count = df_all.groupby(['driver_id', 'start_district_id']).size()
    df_driver_region_count = pd.DataFrame({'driver_id': drivers})

    for i in range(len(drivers)):
        # len(driver_start_count.xs(('driver_id')).values)     # 表示该 driver 接过多少个区域的订单
        try:
            region_num = len(driver_start_count.xs((drivers[i])).values)
            df_driver_region_count.loc[i, 'region_num'] = region_num
        except KeyError:
            print(i)
            print(drivers[i])

    print(df_driver_region_count)
    df_driver_region_count.to_csv('E:\\data\\DiDiData\\data_csv\\driver_region\\drivers_region.csv')


def statistic_driver_region_num():
    df_size = pd.read_csv('E:\\data\\DiDiData\\data_csv\\driver_region\\drivers_region.csv')
    df = pd.DataFrame()
    df1 = df_size.groupby('region_num').size()
    df['region_num'] = df1.index
    df['driver_num'] = df1.values
    df.to_csv('E:\\data\\DiDiData\\data_csv\\driver_region\\drivers_region_num.csv')



if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_data_districtDealed\\'
    # fileList = ft.listdir_nohidden(dirPath)
    # fileList.sort()
    # df = sum_driver_region(fileList)
    statistic_driver_region_num()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



