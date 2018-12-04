# 提取时间特征，包括 date, half_hour, week_day, day, is_weekend, is_vocation
#
# 得到 time_feature.csv，每周平均和历史平均（分工作日和周末）分两部分
# 一个是城市总流量的（totalFlow_30min_replaceTimeBand.csv）
# 一个是每个时间片所有起止对的(order_count_30min\\order_count_1\\)
# 注：count的历史均值取前17天（2.23-3.10， 周二至周四3周，其余两周）的数据

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
from datetime import datetime
from datetime import timedelta


# 添加时间的特征
def add_time_features():
    file_path = 'E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv'
    df = pd.read_csv(file_path)

    print(df.dtypes)
    df['week_day'] = pd.to_datetime(df['date']).map(lambda x: x.isoweekday())
    # df['day'] = pd['date'].map(lambda x: (x - datetime(2016, 2, 22)).days)
    df['day'] = pd.to_datetime(df['date']).map(lambda x: (x - datetime(2016, 2, 22)).days)
    df['is_weekend'] = df['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)
    df['is_vocation'] = df['date'].map(lambda x: 1 if x in pd.date_range('2016/03/08', '2016/03/08') else 0)

    # 前1~3个时间片的的count，lagging_1, lagging_2, lagging_3，前三个时间片的lagging就不变吧
    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'lagging_3'] = df.loc[i, 'count']
            df.loc[i, 'lagging_2'] = df.loc[i, 'count']
            df.loc[i, 'lagging_1'] = df.loc[i, 'count']
        elif i == 1:
            df.loc[i, 'lagging_3'] = df.loc[i - 1, 'count']
            df.loc[i, 'lagging_2'] = df.loc[i - 1, 'count']
            df.loc[i, 'lagging_1'] = df.loc[i - 1, 'count']
        elif i == 2:
            df.loc[i, 'lagging_3'] = df.loc[i - 2, 'count']
            df.loc[i, 'lagging_2'] = df.loc[i - 2, 'count']
            df.loc[i, 'lagging_1'] = df.loc[i - 1, 'count']
        else:
            df.loc[i, 'lagging_3'] = df.loc[i - 3, 'count']
            df.loc[i, 'lagging_2'] = df.loc[i - 2, 'count']
            df.loc[i, 'lagging_1'] = df.loc[i - 1, 'count']

    # print(df.head())
    df = df.rename(columns={'half_hour': 'time'})
    df = df[['date', 'time', 'week_day', 'day', 'is_weekend', 'is_vocation',
             'lagging_3', 'lagging_2', 'lagging_1', 'count']]
    df.set_index(keys=['date', 'time'], inplace=True)

    dest_path = 'E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv'
    df.to_csv(dest_path)


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    add_time_features()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



