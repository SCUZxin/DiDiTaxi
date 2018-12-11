# 在对城市总流量进行预测之前进行归一化处理[0, 1]

import pandas as pd
import numpy as np
from datetime import datetime
import sklearn.preprocessing as preprocessing


def gen_total_set():
    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    # df_time = df_time[['week_day', 'day', 'is_weekend', 'is_vocation', 'lagging_3', 'lagging_2', 'lagging_1', 'count']]
    del df_time['date']
    del df_time['time']
    df = pd.concat([df_weather, df_time], axis=1)
    x, y = df.iloc[:, 1:-1], df.loc[:, 'count']
    return x, y


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # x, y = gen_total_set()
    # x_train, y_train = x.iloc[0:int(len(x)/24*17), :], y[0:int(len(y)/24*17)]
    # x_test, y_test = x.iloc[int(len(x)/24*17):len(x), :], y[int(len(y)/24*17):len(y)]

    import model.gbrt_pair_predict as gbrt_pair_predict
    train, test = gbrt_pair_predict.gen_data_set()
    x_train, y_train = train.iloc[:, 0:-1], train.loc[:, 'count'].values
    x_test, y_test = test.iloc[:, 0:-1], test.loc[:, 'count'].values


    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    x_train_minmax = x_scaler.fit_transform(x_train)
    x_test_minmax = x_scaler.transform(x_test)
    # print(type(y_train.values))
    y_train_minmax = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_minmax = y_scaler.transform(y_test.reshape(-1, 1))
    y_inverse = y_scaler.inverse_transform(y_test_minmax)   # y_inverse = y_test  只是值相等，type()不相等

    print(y_test[0:10])
    print(y_test_minmax[0:10])
    print(list(y_scaler.inverse_transform(y_test_minmax)[0:10]))


    # print(y_test)
    # print(y_test_minmax)
    # print(y_scaler.inverse_transform(y_test_minmax))
    # X_train_minmax = min_max_scaler.fit_transform(X_train)

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


    # X_train = pd.DataFrame({'A': range(11),
    #                         'B': [2,4,6,8,10,12,14,16,18,18,22]})
    # X_test = pd.DataFrame({
    #                         'B': [2,4,6,8,10,12,14,16,18,18,42]})
    #
    # # print(X_train)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform(X_train)
    # # X_train_minmax = min_max_scaler.fit_transform(X_train)
    # print(X_train_minmax)
    #
    # X_test_minmax = min_max_scaler.transform(X_test)
    # print(X_test_minmax)
    #
    # Y = X_test_minmax
    #
    # Y = min_max_scaler.transform(Y.reshape(-1, 1))
    # print(Y)
    #
    # print(min_max_scaler.inverse_transform(Y))


