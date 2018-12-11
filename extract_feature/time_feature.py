# 提取时间特征，包括 date, half_hour, week_day, day, is_weekend, is_vocation
#
# 得到 time_feature.csv，每周平均和历史平均（分工作日和周末）分两部分
# 一个是城市总流量的（totalFlow_30min_replaceTimeBand.csv）
# 一个是每个时间片所有起止对的(order_count_30min\\order_count_1\\)
# 注：count的历史均值取前17天（2.23-3.10， 周二至周四3周，其余两周）的数据

import pandas as pd
import numpy as np
from datetime import datetime


# 添加时间的特征
def add_time_features(df):

    df['day'] = pd.to_datetime(df['date']).map(lambda x: (x - datetime(2016, 2, 22)).days)
    # df['day'] = pd['date'].map(lambda x: (x - datetime(2016, 2, 22)).days)
    df['week_day'] = pd.to_datetime(df['date']).map(lambda x: x.isoweekday())
    df['Monday'] = df.week_day.map(lambda x: 1 if x == 1 else 0)
    df['Tuesday'] = df.week_day.map(lambda x: 1 if x == 2 else 0)
    df['Wednesday'] = df.week_day.map(lambda x: 1 if x == 3 else 0)
    df['Thursday'] = df.week_day.map(lambda x: 1 if x == 4 else 0)
    df['Friday'] = df.week_day.map(lambda x: 1 if x == 5 else 0)
    df['Saturday'] = df.week_day.map(lambda x: 1 if x == 6 else 0)
    df['Sunday'] = df.week_day.map(lambda x: 1 if x == 7 else 0)

    df['is_weekend'] = df['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)
    df['is_vocation'] = df['date'].map(lambda x: 1 if x in pd.date_range('2016/03/08', '2016/03/08') else 0)

    flow_total_np = np.zeros(25 * 48)  # 24 = 24+1（2.22）
    flow_total_np[48:] = np.array(df['count'])
    flow_total_np[0:48] = flow_total_np[48:96]
    count_array = flow_total_np.reshape(-1, 48)

    for i in range(len(df)):
    # for i in range(1):
        time = df.loc[i, 'half_hour']
        day = df.loc[i, 'day']
        # 分别对工作日和周末取平均
        if df.loc[i, 'week_day'] in [6, 7]:
            days = [5, 6, 12, 13]
        else:
            days = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17]
        sum = 0
        # print(count_array[days[0]])
        for j in range(len(days)):
            sum += count_array[days[j], time]
        meanOnHurByDay = sum/len(days)

        # 按照周几取平均，2016.2.22是周一，17天数据，只有周二周三周四是3周，其余是2周
        day = day % 7
        if day == 0:  # 为了不算第一行（2.22）的数据
            days = [7, 14]
        elif day >= 1 and day <= 3:
            days = [day, day + 7, day + 14]
        else:
            days = [day, day + 7]
        sum = 0
        for j in range(len(days)):
            sum += count_array[days[j], time]
        meanOnHurByWeek = sum/len(days)
        # print('meanOnHurByDay: ', meanOnHurByDay)
        # print('meanOnHurByWeek: ', meanOnHurByWeek)
        df.loc[i, 'mean_his_day'] = meanOnHurByDay
        df.loc[i, 'mean_his_week'] = meanOnHurByWeek

    del df['week_day']

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
    df = df[['date', 'time', 'day', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday', 'is_weekend', 'is_vocation','mean_his_day',
             'mean_his_week', 'lagging_3', 'lagging_2', 'lagging_1', 'count']]
    df.set_index(keys=['date', 'time'], inplace=True)

    return df


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    file_path = 'E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_30min_replaceTimeBand.csv'
    df = pd.read_csv(file_path)
    df = add_time_features(df)
    dest_path = 'E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv'
    df.to_csv(dest_path)

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



