# 提取时间特征，包括 date, half_hour, week_day,
#
# 得到 time_feature.csv，每周平均和历史平均（分工作日和周末）分两部分
# 一个是城市总流量的（totalFlow_30min_replaceTimeBand.csv）
# 一个是每个时间片m每个起止对的(order_count_30min\\order_count_1\\)
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

    df['week_day'] = df['date'].map(lambda x: x.isoweekday())
    df['day'] = df['datetime'].map(lambda x: (x - datetime.datetime(2016, 2, 22)).days)

    print(df)


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    add_time_features()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



