# 读取每一天的天气数据文件，进行特征提取，得到weather_feature.csv
# 当天7:00-23:00（14-45时间片，共32个时间片）的所有天气情况取众数

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
from datetime import datetime
from datetime import timedelta


# 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
def weather_statistics(file_list, time_interval_para=30):
    for i in range(len(file_list)):
    # for i in range(1):
    #     i = 14
        file_path = os.path.join(dirPath, file_list[i])
        df = pd.read_csv(file_path)
        df.time = df.time.str.split(':')
        df.time = df.time.map(lambda x: int(x[0]) * 2 + int(x[1]) // time_interval_para)

        size = df.groupby('time').size()
        df_drop_dupli = df.drop_duplicates(['time'])

        # df.loc[(df['weather'] > 4)] = 4   # 用这行代码的话 df_weather_mode 的取众数要出错，我也不知道为什么
        # weather > 4 的天气全部转为4（下雨）
        for j in range(len(df)):
            if df.loc[j, 'weather'] > 4:
                df.loc[j, 'weather'] = 4

        # weather 取众数, groupby('a') 没有 mode()函数
        df_weather_mode = df.groupby('time')['weather'].agg(lambda x: x.value_counts().index[0]).reset_index()
        # 该时间片的天气是否下雨
        is_rain_time_slice = df_weather_mode.weather.map(lambda x: 1 if x == 4 else 0)

        # 当天7:00-23:00（14-45时间片，共32个时间片）的所有天气情况取众数
        # df_weather_of_day_mode = df['weather'].mode()
        df_weather_of_day_mode = df.loc[(df['time'] >= 14) & (df['time'] <= 45)]['weather'].mode()
        # 今天是否下雨
        is_rain_day = 1 if df_weather_of_day_mode[0] == 4 else 0

        temperature_mean = df.groupby('time')['temperature'].mean()
        pm25_mean = df.groupby('time').mean()['pm2.5']

        # 提取 7:00-23:00（14-45时间片，共32个时间片） temperature 和 pm2.5 的平均值
        temp_mean_day = df.loc[(df['time'] >= 14) & (df['time'] <= 45)].temperature.mean()
        pm25_mean_day = df.loc[(df['time'] >= 14) & (df['time'] <= 45)]['pm2.5'].mean()

        df_temp = pd.DataFrame({'date': df.loc[0, 'date'],
                                'time': list(df_drop_dupli.time),
                                'weather': list(df_weather_mode.weather),
                                'is_rain_time_slice': list(is_rain_time_slice),
                                'weather_of_day': df_weather_of_day_mode[0],
                                'is_rain_day': is_rain_day,
                                'temperature': list(temperature_mean),
                                'temp_mean_day': temp_mean_day,
                                'pm2.5': list(pm25_mean),
                                'pm2.5_mean_day': pm25_mean_day})

        df_temp = df_temp[['date', 'time', 'weather', 'is_rain_time_slice', 'weather_of_day',
                           'is_rain_day', 'temperature', 'temp_mean_day', 'pm2.5', 'pm2.5_mean_day']]
        df_temp.set_index(keys=['date', 'time'], inplace=True)

        global df_out
        if i == 0:
            df_out = df_temp
        else:
            # df_out = df_out.append(df_temp)
            df_out = pd.concat([df_out, df_temp], axis=0)

    dest_path = 'E:\\data\\DiDiData\\data_csv\\weather_data\\weather_feature.csv'
    df_out.to_csv(dest_path)


def add_feat():

    # 添加天气信息
    file_path = 'E:\\data\\DiDiData\\data_csv\\weather_data\\weather_feature_without_missing.csv'
    df = pd.read_csv(file_path)
    # 该时间片的天气是否下雨
    df['weather-1'] = df.weather.map(lambda x: 1 if x == 1 else 0)
    df['weather-2'] = df.weather.map(lambda x: 1 if x == 2 else 0)
    df['weather-3'] = df.weather.map(lambda x: 1 if x == 3 else 0)
    df['weather-4'] = df.weather.map(lambda x: 1 if x == 4 else 0)
    df['is_rain_time_slice'] = df.weather.map(lambda x: 1 if x == 4 else 0)
    del df['weather']
    # 今天是否下雨
    df['weather_of_day-1'] = df.weather_of_day.map(lambda x: 1 if x == 1 else 0)
    df['weather_of_day-2'] = df.weather_of_day.map(lambda x: 1 if x == 2 else 0)
    df['weather_of_day-3'] = df.weather_of_day.map(lambda x: 1 if x == 3 else 0)
    df['weather_of_day-4'] = df.weather_of_day.map(lambda x: 1 if x == 4 else 0)
    df['is_rain_day'] = df.weather_of_day.map(lambda x: 1 if x == 4 else 0)
    del df['weather_of_day']

    # 添加 temperature 和 pm2.5 的变化量（和上一个时间片相比）
    # 前一两个时间片的天气
    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'temp_change'] = 0
            df.loc[i, 'pm2.5_change'] = 0
            # df.loc[i, 'weather_pre2'] = 3
            # df.loc[i, 'weather_pre1'] = 3
        # elif i == 1:
        #     df.loc[i, 'temp_change'] = df.loc[i, 'temperature'] - df.loc[i-1, 'temperature']
        #     df.loc[i, 'pm2.5_change'] = df.loc[i, 'pm2.5'] - df.loc[i-1, 'pm2.5']
            # df.loc[i, 'weather_pre2'] = 3
            # df.loc[i, 'weather_pre1'] = df.loc[i-1, 'weather']
        else:
            df.loc[i, 'temp_change'] = df.loc[i, 'temperature'] - df.loc[i-1, 'temperature']
            df.loc[i, 'pm2.5_change'] = df.loc[i, 'pm2.5'] - df.loc[i-1, 'pm2.5']
            # df.loc[i, 'weather_pre2'] = df.loc[i-2, 'weather']
            # df.loc[i, 'weather_pre1'] = df.loc[i-1, 'weather']

    df = df[['date', 'time', 'weather-1', 'weather-2', 'weather-3', 'weather-4', 'is_rain_time_slice',
             'weather_of_day-1', 'weather_of_day-2', 'weather_of_day-3', 'weather_of_day-4', 'is_rain_day',
             'temperature', 'temp_change', 'temp_mean_day', 'pm2.5', 'pm2.5_change', 'pm2.5_mean_day']]
    df.set_index(keys=['date', 'time'], inplace=True)
    df.to_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # dirPath = '/home/zx/data/DiDiData/data_csv/order_lite/'
    dirPath = 'E:\\data\\DiDiData\\data_csv\\weather_data\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()

    # 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
    # weather_statistics(fileList, 30)
    add_feat()

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

