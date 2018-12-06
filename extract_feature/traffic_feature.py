# 提取交通的特征：
# 1.以30min为时间片，分别得到拥堵程度为1-4的平均路段数（取整）
#   难点：有的时间片半小时有3次记录，有的是2次

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
import re
from datetime import datetime

def get_traffic_info(file_list, time_interval_para=30):
    for i in range(len(file_list)):
    # for i in range(1):
        file_path = os.path.join(dirPath, file_list[i])
        df = pd.read_csv(file_path)
        file_date = file_list[i].split('_')[2].split('.')[0]
        df['time'] = df.tj_time.str.split()
        del df['tj_time']
        df['time'] = df.time.map(lambda x: x[1])
        df.time = df.time.str.split(':')
        df.time = df.time.map(lambda x: int(x[0]) * 2 + int(x[1]) // time_interval_para)

        df.tj_level = df.tj_level.str.split('-')
        df['tj_level_1'] = df.tj_level.map(lambda x: int(x[0].split(':')[1]))
        df['tj_level_2'] = df.tj_level.map(lambda x: int(x[1].split(':')[1]))
        df['tj_level_3'] = df.tj_level.map(lambda x: int(x[2].split(':')[1]))
        df['tj_level_4'] = df.tj_level.map(lambda x: int(x[3].split(':')[1]))
        del df['tj_level']

        # print(df.head(5))
        cols = [col for col in df.columns if col not in ['district_id', 'time']]

        df = df.groupby(['district_id', 'time'])[cols].mean().round()
        df['district_id'] = df.index.get_level_values('district_id')
        df['time'] = df.index.get_level_values('time')
        df['date'] = pd.Timestamp(file_date)

        # 没有数据信息的区域id，将其tj_level_1-4都置为0，没有区域13,51
        district_list = set(df['district_id'].values)
        for j in range(1, 59):
            if j not in district_list:
                print(file_date, '  ', j)
                df_tmp = pd.DataFrame({'district_id': j,
                                       'date': pd.Timestamp(file_date),
                                       'time': range(0, 48),
                                       'tj_level_1': 0,
                                       'tj_level_2': 0,
                                       'tj_level_3': 0,
                                       'tj_level_4': 0,
                                       })
                df = df.append(df_tmp)

        df = df[['district_id', 'date', 'time', 'tj_level_1', 'tj_level_2', 'tj_level_3', 'tj_level_4']]
        df = df.sort_values(by=['district_id', 'date', 'time'], axis=0, ascending=True)
        df.set_index(keys=['district_id', 'date', 'time'], inplace=True)

        if i == 0:
            df_traffic = df
        else:
            df_traffic = pd.concat([df_traffic, df], axis=0)

        df_traffic.to_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    dirPath = 'E:\\data\\DiDiData\\data_csv\\traffic_data_districtDealed\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    # 以30min为时间间隔，将30min内的 1-4级的交通情况 提取总结出来，并写入 traffic.csv文件
    get_traffic_info(fileList, 30)

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

