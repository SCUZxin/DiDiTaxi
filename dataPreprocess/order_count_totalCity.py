# 以 t 分钟为时间片（10min，15min，30min）, 统计整个城市每个时间片总的订单量
# 去除目的地id为-1的数据

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta


def flow_statistics_totalCity(fileList, time_interval = 15, threshold = 0):
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
        del df['Time']

        order_date = date(year, month, day)
        df['Date'] = np.array(order_date)   # df['Date'] = pd.Timestamp(fileDate)

        starttime = datetime(year, month, day, 00, 00, 00)
        bins = [starttime + i * time_interval for i in range(1 + int(24 * 60 * 60 / time_interval.seconds))]  # 取值区间（]

        df['TimeBand'] = pd.cut(times, bins)
        df['TimeBand'] = df['TimeBand'].astype(str) #强制类型转换，避免append时 categorical类型转换错误

        # 某时间片整个城市的 order_count
        df = df.groupby(['Date', 'TimeBand']).size()

        df = pd.DataFrame(df)
        df.rename(columns={0: 'count'}, inplace=True)

        # 排序后 (2016-02-23, 2016-02-23 00:15:00] 到最后一行了，切片，将最后一行移至第一行
        length = len(df)
        df = df.iloc[length - 1:, :].append(df.iloc[0:length - 1])

        # print(df.dtypes)

        if i == 0:
            df_out = df
        else:
            # df_out = df_out.append(df, ignore_index=True)
            df_out = df_out.append(df)
        print(fileDate)

        # 最后一天的数据加入后
        if i == len(fileList) - 1:
        # if i == 1:
            # df_out = df_out.unstack(0)
            destPath = 'E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_60min.csv'
            df_out.to_csv(destPath)


if __name__ == "__main__":

    time_Now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', time_Now)
    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_lite\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()

    time_interval = timedelta(minutes=60)
    threshold = 0
    # 调用方法获取满足时间间隔和阈值的 order_count_totalCity , 并写入 csv文件
    flow_statistics_totalCity(fileList, time_interval, threshold)



