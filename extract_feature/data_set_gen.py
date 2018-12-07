# 产生起止对pair数据集（包括训练集2.23-3.10和测试集3.11-3.17）

import numpy as np
import pandas as pd

import Tools.funcTools as ft
import os
from datetime import datetime
import pickle

import extract_feature.poi_level_first_count as poi_count_file


# 生成的pkl文件包含的内容是dict
# key：起始地和目的地的id字符串相加， value：该起点终点对的所有订单量按时间排序形成的订单量向量
def GetEveryPairData(df):
    print('start GetEveryPairData(df) method')
    try:
        D = pickle.load(open('E:\\data\\DiDiData\\data_csv\\dataset\\sd_pair.pkl', 'rb'))
    except FileNotFoundError:
        print('FileNotFoundError')
        D = {}
        g = df.groupby(['start_district_id', 'dest_district_id', 'date', 'time']).sum()
        indexs = g.index    # <class 'pandas.core.indexes.multi.MultiIndex'>  g.index[4]=(1, 1, '2016-02-23', 37)
        count_list = g['count'].values
        i = 0
        # print(len(count_list))   # 385412
        while i < len(count_list):
            s, d = indexs[i][0], indexs[i][1]     # s：start_district_id， t：dest_district_id
            if i % 10000 >= 0 and i % 10000 < 100:  # 刚好i是10000的倍数比较少
                print('%s gen %d neighboors' % (datetime.now(), i))
            # 保存的是距离(2016,2,22)的小时的时间片(index)的订单量count值(value),起点和终点已经确定是s和d了
            tmp = np.zeros(25*48)   # 19 = 17+1（2.22）+1（3.11）
            j = i
            # 前提是要按照['start_district_id','dest_district_id','date','time'] 排序
            while j < len(count_list) and indexs[j][0] == s and indexs[j][1] == d:
                tmp[(datetime.strptime(indexs[j][2], '%Y-%m-%d')-datetime(2016, 2, 22)).days * 48
                    + indexs[j][3]] = count_list[j]
                j = j + 1
            tmp[0:48] = tmp[48:96]      # 2.22 <- 2.23
            # tmp[-48:] = tmp[-96:-48]    # 3.11 <- 3.10
            D[str(s)+'-'+str(d)] = tmp # D[s+t]是字符串相加，自己用的时候注意（可改为1-2,52-58等）
            i = j
        pickle.dump(D, open('E:\\data\\DiDiData\\data_csv\\dataset\\sd_pair.pkl', 'wb'))

    print('end GetEveryPairData(df) method')
    return D


# 根据参数决定返回所有数据集、训练集 or 测试集，对应的参数是 all、train、test
def order_count_contact(target='all'):
    print('start order_count_contact() method')
    df_dataSet = pd.DataFrame()
    try:
        df_dataSet = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\'+target+'set.csv')
    except FileNotFoundError:
        dir_path = 'E:\\data\\DiDiData\\data_csv\\order_count_30min\\order_count_replaceTimeBand\\'
        file_list = ft.listdir_nohidden(dir_path)
        file_list.sort()

        start = 0
        end = len(file_list)
        if target == 'train':
            end = 17
        elif target == 'test':
            start = 17

        for i in range(start, end):
            file_path = os.path.join(dir_path, file_list[i])
            df = pd.read_csv(file_path)
            df.rename(columns={'half_hour': 'time'}, inplace=True)
            df.set_index(keys=['start_district_id', 'dest_district_id', 'date', 'time'], inplace=True)
            if i == start:
                df_dataSet = df
            else:
                df_dataSet = pd.concat([df_dataSet, df], axis=0)

    print('end order_count_contact() method')
    return df_dataSet


# 将时间、天气、交通、POI等特征信息集成在一起
def gen_basic_feature(df):
    # df = df.loc[112728:112735]

    # df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')
    df_traffic = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
    poi_count_dict = poi_count_file.poi_count()

    # 添加时间特征
    df['day'] = pd.to_datetime(df['date']).map(lambda x: (x - datetime(2016, 2, 22)).days)
    df['week_day'] = pd.to_datetime(df['date']).map(lambda x: x.isoweekday())
    df['is_weekend'] = df['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)
    df['is_vocation'] = df['date'].map(lambda x: 1 if x in pd.date_range('2016/03/08', '2016/03/08') else 0)

    print('------------------------------')

    time_feature = []
    weather_feature = []
    traffic_feature = []
    poi_start_feature = []
    poi_dest_feature = []
    for i in range(len(df)):
        # i += 112728
        if i % 10000 == 0:
            print('iterator %d' % i)
        s = df.loc[i, 'start_district_id']
        d = df.loc[i, 'dest_district_id']
        date = datetime.strptime(df.loc[i, 'date'], '%Y-%m-%d')
        time = df.loc[i, 'time']
        time_feature.append(add_time_feature(df, s, d, date, time, i))

        # 添加天气特征
        weather = list(df_weather.loc[(pd.to_datetime(df_weather['date']) == date)
                                  & (df_weather['time'] == time)].iloc[0, 2:])
        weather_feature.append(weather)

        # 添加起始地的交通特征
        traffic = list(df_traffic.loc[(df_traffic['district_id'] == s)
                    & (pd.to_datetime(df_traffic['date']) == date) & (df_traffic['time'] == time)].iloc[0, 3:])
        traffic_feature.append(traffic)

        # 添加起始地、目的地的POI特征
        poi_start = list(poi_count_dict[str(s)])
        poi_dest = list(poi_count_dict[str(d)])
        poi_start_feature.append(poi_start)
        poi_dest_feature.append(poi_dest)

        # break

    df_time = pd.DataFrame(np.array(time_feature),
                           columns=['mean_his_day', 'mean_his_week', 'lagging_3', 'lagging_2', 'lagging_1'])
    columns_weather = df_weather.columns[2:]
    df_weather = pd.DataFrame(np.array(weather_feature), columns=columns_weather)
    columns_traffic = df_traffic.columns[3:]
    df_traffic = pd.DataFrame(np.array(traffic_feature), columns=columns_traffic)
    columns_start_poi = ['start_poi_'+str(i)+'_count' for i in range(1, 26)]
    df_start_poi = pd.DataFrame(np.array(poi_start_feature), columns=columns_start_poi)
    columns_dest_poi = ['dest_poi_'+str(i)+'_count' for i in range(1, 26)]
    df_dest_poi = pd.DataFrame(np.array(poi_dest_feature), columns=columns_dest_poi)

    # df_time = pd.DataFrame(np.array(time_feature),index=range(112728, 112728+len(df)),
    #                        columns=['mean_his_day', 'mean_his_week', 'lagging_3', 'lagging_2', 'lagging_1'])
    # df_weather = pd.DataFrame(np.array(weather_feature), columns=columns_weather, index=range(112728, 112728+len(df)))
    # df_traffic = pd.DataFrame(np.array(traffic_feature), columns=columns_traffic, index=range(112728, 112728+len(df)))
    # df_start_poi = pd.DataFrame(np.array(poi_start_feature), columns=columns_start_poi, index=range(112728, 112728+len(df)))
    # df_dest_poi = pd.DataFrame(np.array(poi_dest_feature), columns=columns_dest_poi, index=range(112728, 112728+len(df)))

    # print(df.columns)
    Tset = pd.concat([df.iloc[:, 0:4], df_weather,df_traffic, df_start_poi, df_dest_poi,
                      df.iloc[:, 5:9], df_time, df['count']], axis=1)
    Tset = Tset.sort_values(by=['start_district_id', 'dest_district_id', 'date', 'time'], axis=0, ascending=True)
    return Tset


def add_time_feature(df, s, d, date, time, i):
    key = str(s) + '-' + str(d)
    if key in sd_pair_dict:
        count_list = sd_pair_dict[key]
        pos = (date - datetime(2016, 2, 22)).days * 48 + time
        ind = pos + np.array(range(-3, 0))
        lagging_list = list(count_list[ind])
        # half_hour = df.loc[i, 'time']
        days = (date - datetime(2016, 2, 22)).days
        r = count_list.reshape((-1, 48))  # reshape成48列，行数不限

        # 分别对工作日和周末取平均
        if df.loc[i, 'is_weekend'] == 1:
            meanOnHurByDay = r[[5, 6, 12, 13], time].mean()
            print('meanOnHurByDay  ', meanOnHurByDay)
        else:
            meanOnHurByDay = r[[1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17], time].mean()

        # 按照周几取平均，2016.2.22是周一，17天数据，只有周二周三周四是3周，其余是2周
        if days % 7 == 0:     # 为了不算第一行（2.22）的数据
            # meanOnHurByWeek_1 = r[7:18:7, half_hour].sum() / 2
            meanOnHurByWeek = r[7:18:7, time].mean()
        else:
            # meanOnHurByWeek = r[days % 7:18:7, half_hour].sum() / (5 + (days % 7 >= 1 and days % 7 <= 3))
            meanOnHurByWeek = r[days % 7:18:7, time].mean()
        return [meanOnHurByDay, meanOnHurByWeek] + lagging_list
    else:
        lagging_list = [0, 0, 0]
        return [0, 0] + lagging_list


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    df_train_set = order_count_contact('train')
    df_test_set = order_count_contact('test')
    df_data_set = order_count_contact('all')

    sd_pair_dict = GetEveryPairData(df_data_set)

    df_train = gen_basic_feature(df_train_set)
    df_test = gen_basic_feature(df_test_set)

    df_train.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Train.csv', index=False)
    df_test.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv', index=False)

    # df_train_set.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\trainset.csv')
    # df_data_set.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    # df_test_set.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\testset.csv')

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

