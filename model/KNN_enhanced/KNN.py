# 对比算法：enhanced KNN
# 思路：
# 1. 构建出所有OD对的历史序列数据，没有数据的要以0代替，得到所有OD对的订单序列
# 2. lag duration = 6，使用加权的欧氏距离找到与预测时间片前6个时间片的值组成的序列最相近的 K=10 个序列
# 3. 对这 K=10 个序列 按照相似度进行排序，根据 ranking 计算相应的权重 Wi
# 3. 对这 K=10 个序列得到后一个值，将其取出进行 Wi 加权得到预测值


from math import sqrt
import numpy as np
from numpy.linalg import *
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# from pyramid.arima import auto_arima
from statsmodels.tsa.ar_model import AR
from sklearn.decomposition import NMF

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import pickle


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i]-y_predict[i]) / y_real[i]
    return sum1/len(y_predict)


# 获取每个OD对的序列 key：OD对（eg：45-58），只需要找数据中有的OD对，value：序列list  len=1152
def get_od_sequence_dict():
    try:
        od_sequence_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\result\\KNN\\od_sequence_dict.pkl', 'rb'))
    except FileNotFoundError:
        df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\KNN\\allset_ordered.csv')
        df_set['day'] = pd.to_datetime(df_set['date']).map(lambda x: (x - datetime(2016, 2, 23)).days)
        len_df = len(df_set)
        # df_set = df_set.groupby(['start_district_id', 'dest_district_id', 'time'])['count'].sum()
        # df = pd.DataFrame()
        # df['start_district_id'] = df_set.index.get_level_values('start_district_id')
        # df['dest_district_id'] = df_set.index.get_level_values('dest_district_id')
        # df['time'] = df_set.index.get_level_values('time')
        # df['count'] = list(df_set.values)

        # 得到每一个OD对的订单量的 list ，保存在 od_sequence_dict.pkl
        od_sequence_dict = {}
        s = 0
        d = 0
        for i in range(len_df):
            if i % 10000 == 0:
                print('iterator: ', i)
            # 获取这一行的所有值 start_district_id, dest_district_id, date, time, count, day
            value = list(df_set.loc[i].values)
            # 第一行，直接赋值并继续循环
            if i == 0:
                sequence_list = [0]*1152
                s = value[0]
                d = value[1]
                sequence_list[value[5] * 48 + value[3]] = value[4]
                continue

            # 最后一行，起止地没有变化，赋值后继续
            if i == len_df - 1:
                sequence_list[value[5] * 48 + value[3]] = value[4]
                od_sequence_dict[str(s) + '-' + str(d)] = sequence_list
                continue

            # 起止地相同，赋值之后直接跳出
            if s == value[0] and d == value[1]:
                sequence_list[value[5] * 48 + value[3]] = value[4]
            # 起止地不同，s,d重新赋值，初始化 sequence_list 并第一次赋值
            else:
                od_sequence_dict[str(s)+'-'+str(d)] = sequence_list
                s = value[0]
                d = value[1]
                sequence_list = [0]*1152
                sequence_list[value[5] * 48 + value[3]] = value[4]
        pickle.dump(od_sequence_dict, open('E:\\data\\DiDiData\\data_csv\\result\\KNN\\od_sequence_dict.pkl', 'wb'))
    return od_sequence_dict


# 使用加权的KNN找出相似度最高的序列并排序, 要预测的 index 是从 816-1151
def predict_batch():
    df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    # df_set['date'] = pd.to_datetime(df_set['date']).reset_index(drop=True)
    df_set['day'] = pd.to_datetime(df_set['date']).map(lambda x: (x - datetime(2016, 2, 23)).days)
    df_test = df_set.loc[385412:, :].reset_index(drop=True)
    # 对df_test 的每一个进行计算预测？？？
    y_predict = []
    for i in range(len(df_test)):
        if i % 10000 == 0:
            print('iterator: ', i)
        # 获取这一行的所有值 start_district_id, dest_district_id, date, time, count, day
        value = list(df_test.loc[i].values)
        y = predict_single(start=value[0], dest=value[1], tar_pos=value[5]*48+value[3])
        y_predict.append(y)

    print(len(df_test))
    print(len(y_predict))
    y_test = list(df_test['count'].values)
    y_predict = list(map(lambda x: round(x), y_predict))
    # 如果遇到负数转为正数，免得计算MSLE出错
    y_predict = list(map(lambda x: -x if x < 0 else x, y_predict))
    df_test['predict_count'] = y_predict
    df_test.to_csv('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_result.csv')

    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    mape = mean_absolute_perc_error(y_predict, y_test)
    print("MAPE: %.4f" % mape)  # 输出平均百分比绝对误差
    me = max(list(map(lambda x1,x2:abs(x1-x2), y_predict, y_test)))
    print("ME: %.4f" % me)  # 输出最大误差
    msle = mean_squared_log_error(y_test, y_predict)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    with open('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_para.txt', 'a') as f:
        f.write('%.4f' % mse+'      %.4f' % mae+'       %.4f' % msle+'      %f' % r2+'\n')


def predict_single(start=9, dest=3, tar_pos=816):
    global od_sequence_dict, lag, K
    sequence = od_sequence_dict[str(start)+'-'+str(dest)]
    subject_list = np.asarray(sequence[tar_pos-lag:tar_pos])  # 最后一个是用来预测的
    ind = tar_pos
    candidate_list = []
    while ind >= (48+lag):
        ind = ind - 48
        temp_list = sequence[ind - lag:ind+1]
        candidate_list.append(temp_list)

    # 计算这些 list 和 subject_list 的相似度，然后排序得到前10个
    dists = []
    for i in range(len(candidate_list)):
        dists.append(norm(np.asarray(candidate_list[i][:-1]) - subject_list))
    dists_ordered = sorted(dists)

    ranks = {}
    for i in range(len(dists)):
        rank = dists_ordered.index(dists[i])+1   # 得到排序
        if rank <= K:
            ranks[str(i)] = rank        # key：rank前10的index，value：rank
    values = list(ranks.values())
    sum1 = 0
    for i in range(len(values)):
        sum1 += (K-values[i]+1)**2
    predict_y = 0
    # 计算每一个候选序列的权重，加权得到预测值
    for key in ranks.keys():
        weight = (K-ranks[key]+1)**2 / sum1  # 得到权值 key：rank前10的index，value：权值
        predict_y += weight * candidate_list[int(key)][-1]
    return predict_y


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    od_sequence_dict = get_od_sequence_dict()
    K = 3
    lag = 6
    predict_batch()
    # for K in range(2, 10):
    #     for lag in range(2, 10):
    #         with open('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_para.txt', 'a') as f:
    #             f.write('K='+ str(K) + '  lag=' + str(lag) + ' :\n')
    #         predict_batch()

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# K=10, lag=6
# MSE: 64.4384
# MAE: 3.2571
# MSLE: 0.3377
# r^2 on test data : 0.960201

# K=3， lag=6
# MSE: 68.9234
# MAE: 3.4995
# MSLE: 0.3890
# r^2 on test data : 0.957431



