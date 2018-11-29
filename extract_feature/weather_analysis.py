# 分析天气数据的分布，各个天气下订单量的多少等

import pandas as pd
import numpy as np
from scipy import stats
import Tools.funcTools as ft
import os
from datetime import datetime
from datetime import timedelta


# 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
def weather_statistics(file_list, time_interval_para=30):
    # for i in range(len(file_list)):
    for i in range(1):
        file_path = os.path.join(dirPath, file_list[i])
        df = pd.read_csv(file_path)
        df.time = df.time.str.split(':')
        df.time = df.time.map(lambda x: int(x[0]) * 2 + int(x[1]) // time_interval_para)
        size = df.groupby('time').size()
        # weather 取众数, groupby('a') 没有 mode()函数
        df_drop_dupli = df.drop_duplicates(['time'])
        df_weather_mode = df.groupby('time')['weather'].agg(lambda x: x.value_counts().index[0]).reset_index()
        temperature_mean = df.groupby('time')['temperature'].mean()
        pm25_mean = df.groupby('time').mean()['pm2.5']

        # print(df_drop_dupli)

        df.time = df_drop_dupli.time
        df.weather = df_weather_mode.weather
        df.temperature = temperature_mean
        df['pm2.5'] = pm25_mean
        print(df)
        # print(temperature_mean)
        # print(pm25_mean)


        # df.Time = pd.Series(pd.to_datetime(df['Time']))  # 将时间的类型由str转为datetime64
        # df = df.sort_values(by='Time', axis=0, ascending=True)
        # df.set_index(keys='order_id', inplace=True)

        # 根据bins将数据times, 以时间间隔为time_interval划分
        # time = np.array(df.Time)

        # starttime = datetime.time(00, 00, 00)
        # # time_interval = timedelta(minutes=15)
        # bins = [starttime + i * time_interval_para for i in range(1 + int(24 * 60 * 60 / time_interval_para.seconds))]  # 取值区间（]

        # df['TimeBand'] = pd.cut(times, bins)
        # del df['Time']

        # 某时间片某区域所有的 order_count
        # df = df.groupby(['TimeBand','start_district_id']).size()       # size = list(df.size())
        # 某时间片某区域到另一区域的 order_count
        # df = df.groupby(['TimeBand', 'start_district_id', 'dest_district_id']).size()

        # print(df.values)
        # print(df.index())

        # df = pd.DataFrame(df)
        # df.rename(columns={0: 'count'}, inplace=True)
        #
        # dest_path = 'E:\\data\\DiDiData\\data_csv\\order_count_15min\\order_count_' + file_date + '.csv'
        # destPath = '/home/zx/data/DiDiData/data_csv/flow_statistics/flow_' + fileDate + '.csv'

        # df.to_csv(dest_path)
        #
        # print(file_date)


if __name__ == '__main__':

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start time:', start_time)

    # dirPath = '/home/zx/data/DiDiData/data_csv/order_lite/'
    dirPath = 'E:\\data\\DiDiData\\data_csv\\weather_data\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()

    # 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
    weather_statistics(fileList, 30)
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end time:', end_time)

