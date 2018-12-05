# 产生起止对pair数据集（包括训练集2.23-3.10和测试集3.11-3.17）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Tools.funcTools as ft
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime
import pickle


def GetEveryPairData(df):
    print('start GetEveryPairData(df) method')
    try:
        D = pickle.load(open('sd_pair.pkl', 'rb'))
    except FileNotFoundError:
        print('FileNotFoundError')
        # D：key：起始地和目的地的id字符串相加， value：该起点终点对的所有订单量按时间排序形成的订单量向量
        D = {}
        g = df.groupby(['start_district_id', 'dest_district_id', 'date', 'time']).sum()
        print(g)
        print(g.columns)
        # print(g.size().values)
        count_list = g['count'].values
        print(count_list)
        print(g.values)
        return
        mat = count_list
        i = 0
        # print(g.head(10))
        # print(len(mat))   # 556168
        while i < len(mat):
            s,t = mat[i,0],mat[i,1]     # s：start_geo_id， t：end_geo_id
            if i%10000==0:
                print('%s gen %d neighboors\n'%(dt.datetime.now(),i))
            # 保存的是距离(2017,6,30)的小时的时间片(index)的订单量count值(value),起点和终点已经确定是s和t了
            tmp = np.zeros(40*24)   # 40 = 31（7月）+7（8月）+1（6.30）+1（8.8）
            j = i
            # 前提是要按照['start_geo_id','end_geo_id','create_date','create_hour'] 排序
            while j<len(mat) and mat[j,0]==s and mat[j,1]==t:
                tmp[ (datetime.strptime( mat[j,2] ,'%Y-%m-%d' )-dt.datetime(2017,6,30)).days * 24 + mat[j,3] ] = mat[j,-1]
                j = j + 1
            tmp[0:24] = tmp[24:48]  #6.30 <- 7.1
            tmp[-24:] = tmp[-48:-24]# 8.8 <- 8.7
            D[s+t] = tmp # D[s+t]是字符串相加，自己用的时候注意（可改为1-2,52-58等）
            i=j
        pickle.dump(D,open('EveryPair.pkl','wb'))   # dict转存为 .pkl文件，能不能存为 .csv文件？
    return D


# 根据参数决定返回所有数据集、训练集 or 测试集，对应的参数是 all、train、test
def order_count_contact(target='all'):
    print('start order_count_contact() method')
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


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_train_set = order_count_contact('train')
    sd_pair = GetEveryPairData(df_train_set)

    # df_train_set.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\trainset.csv')

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

