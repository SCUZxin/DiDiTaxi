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
            tmp = np.zeros(18*48)   # 19 = 17+1（2.22）+1（3.11）
            j = i
            # 前提是要按照['start_district_id','dest_district_id','date','time'] 排序
            while j<len(count_list) and indexs[j][0]==s and indexs[j][1]==d:
                tmp[(datetime.strptime(indexs[j][2], '%Y-%m-%d')-datetime(2016,2,22)).days * 48 + indexs[j][3]] = count_list[j]
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
def gen_basic_feature(df, sd_pair_dict):
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')
    df_traffic = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
    poi_count_dict = poi_count_file.poi_count()
    print('======')

    # 添加时间特征
    df['is_vocation'] = pd.to_datetime(df['date']).map(lambda x: 1 if x in pd.date_range('2016/03/08', '2016/03/08') else 0)
    df['week_day'] = pd.to_datetime(df['date']).map(lambda x: x.isoweekday())
    df['day'] = pd.to_datetime(df['date']).map(lambda x: (x - datetime(2016, 2, 22)).days)
    df['is_weekend'] = df['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)

    print(df.head(10))
    feature_list = []
    for i in range(len(df)):
        if i % 10000 == 0:
            print('iterator %d' % i)
        s = df.loc[i, 'start_district_id']
        d = df.loc[i, 'dest_district_id']
        add_feature(sd_pair_dict, s, d, i)
        # feature_list.append(add_feature(ep, mat[i, 0], mat[i, 1], mat[i, 2], mat[i, 3]))



    print(df.head())
    return df


def add_feature(df, s, d, i):

    key = str(s) + '-' + str(d)
    if key in sd_pair_dict:
        count_list = sd_pair_dict[key]
        pos = (datetime.strptime(df.loc[i, 'date'], '%Y-%m-%d') - datetime(2016, 2, 22)).days * 48 + df.loc[i, 'time']
        df.loc[i, 'lagging_3'] = count_list[pos - 3]
        df.loc[i, 'lagging_2'] = count_list[pos - 2]
        df.loc[i, 'lagging_1'] = count_list[pos - 1]
    else:
        df.loc[i, 'lagging_3'] = 0
        df.loc[i, 'lagging_2'] = 0
        df.loc[i, 'lagging_1'] = 0


    half_hour = df.loc[i, 'time']
    days = (datetime.strptime(df.loc[i, 'date'], '%Y-%m-%d') - datetime(2017, 6, 30)).days
    r = sd_pair_dict.reshape((-1, 48))  # reshape成48列，行数不限
    meanOnHurByDay = r[1:-1, half_hour].sum() / (34 + (half_hour + 1) % 2)
    # 2017.6.30是周五，38天数据，只有周六周日周一是6周，其余是5周，周平均数据,原代码好像有逻辑问题
    if (days % 7 == 0):
        meanOnHurByWeek = r[7:-1:7, half_hour].sum() / (5 + (days % 7 >= 1 and days % 7 <= 3))
    else:
        meanOnHurByWeek = r[days % 7:-1:7, half_hour].sum() / (5 + (days % 7 >= 1 and days % 7 <= 3))


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_train_set = order_count_contact('train')
    df_test_set = order_count_contact('test')
    sd_pair_dict = GetEveryPairData(df_train_set)
    # train, test = SSSet()
    df_train = gen_basic_feature(df_train_set, sd_pair_dict)

    # df_train_set.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\trainset.csv')

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

