# 以 t 分钟为时间间隔, 统计在每个时间片, 区域 i 到 区域 j 的订单量（order_count）,
# 去除目的地id为-1的数据

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
from datetime import datetime
from datetime import timedelta

# 时间间隔默认为30分钟, 订单量阈值默认为0（低于该阈值的区域订单量不保留）,订单量统计并写入文件
# 阈值筛选还没做（进行订单量预测功能，不做阈值筛选，可在后期根据订单量级对区域分类之后再进行预测）
def new_flow_statistics(fileList, time_interval = 30, threshold = 0):
    for i in range(len(fileList)):
    # for i in range(1):
        filePath = os.path.join(dirPath, fileList[i])

        fileDate = fileList[i].split('_')[1].split('.')[0]  # 2016-02-23
        dateList = fileDate.split('-')
        year = int(dateList[0])
        month = int(dateList[1])
        day = int(dateList[2])

        df = pd.read_csv(filePath)
        df = df[df.dest_district_id != -1]  # 去除目的地为-1的行数据

        df.Time = pd.Series(pd.to_datetime(df['Time']))  # 将时间的类型由str转为datetime64
        df = df.sort_values(by='Time', axis=0, ascending=True)
        df.set_index(keys='order_id', inplace=True)

        # 根据bins将数据times, 以时间间隔为time_interval划分
        times = np.array(df.Time)

        starttime = datetime(year, month, day, 00, 00, 00)
        # time_interval = timedelta(minutes=15)
        bins = [starttime + i * time_interval for i in range(1 + int(24 * 60 * 60 / time_interval.seconds))]  # 取值区间（]

        df['TimeBand'] = pd.cut(times, bins)
        del df['Time']

        # 某时间片某区域所有的 order_count
        # df = df.groupby(['TimeBand','start_district_id']).size()       # size = list(df.size())
        # 某时间片某区域到另一区域的 order_count
        df = df.groupby(['TimeBand', 'start_district_id', 'dest_district_id']).size()

        # print(df.values)
        # print(df.index())

        df = pd.DataFrame(df)
        df.rename(columns={0: 'count'}, inplace=True)

        destPath = 'E:\\data\\DiDiData\\data_csv\\order_count_15min\\order_count_' + fileDate + '.csv'
        # destPath = '/home/zx/data/DiDiData/data_csv/flow_statistics/flow_' + fileDate + '.csv'

        df.to_csv(destPath)

        print(fileDate)


if __name__ == '__main__':

    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', time)

    # dirPath = '/home/zx/data/DiDiData/data_csv/order_lite/'
    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_lite\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()

    time_interval = timedelta(minutes=15)
    threshold = 0
    # 调用方法获取满足时间间隔和阈值的 order_count , 并写入 csv文件
    new_flow_statistics(fileList, time_interval, threshold)




