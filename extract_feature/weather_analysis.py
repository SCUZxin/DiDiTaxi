# 分析天气数据的分布，各个天气下订单量的多少等?取影响较大的区域，而不是整体流量？ 这点还没做

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
        file_path = os.path.join(dirPath, file_list[i])
        df = pd.read_csv(file_path)
        df.time = df.time.str.split(':')
        df.time = df.time.map(lambda x: int(x[0]) * 2 + int(x[1]) // time_interval_para)

        # size = df.groupby('time').size()
        # weather 取众数, groupby('a') 没有 mode()函数
        df_drop_dupli = df.drop_duplicates(['time'])
        df_weather_mode = df.groupby('time')['weather'].agg(lambda x: x.value_counts().index[0]).reset_index()
        df_weather_of_day_mode = df['weather'].mode()   # 当天的所有天气情况取众数
        temperature_mean = df.groupby('time')['temperature'].mean()
        pm25_mean = df.groupby('time').mean()['pm2.5']

        df_temp = pd.DataFrame({'date': df.loc[0, 'date'],
                                'time': list(df_drop_dupli.time),
                                'weather': list(df_weather_mode.weather),
                                'weather_of_day': df_weather_of_day_mode[0],
                                'temperature': list(temperature_mean),
                                'pm2.5': list(pm25_mean)})

        df_temp = df_temp[['date', 'time', 'weather', 'weather_of_day', 'temperature', 'pm2.5']]
        df_temp.set_index(keys=['date', 'time'], inplace=True)

        global df_out
        if i == 0:
            df_out = df_temp
        else:
            # df_out = df_out.append(df_temp)
            df_out = pd.concat([df_out, df_temp], axis=0)

        print(len(df_out))

    dest_path = 'E:\\data\\DiDiData\\data_csv\\weather_data\\weather_feature.csv'
    df_out.to_csv(dest_path)


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

