# 使用 HA(历史平均) 模型预测所有时间片上起止对的订单量
# mean_his_day: 区分工作日和周末取平均
# mean_his_week: 按照周几取平均，2016.2.22是周一，17天数据，只有周二周三周四是3周，其余是2周
# 按照时间排序，如果需要按照起止对排序的话，使用 df = df.sort_values(...)

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from datetime import datetime


def get_mean_his():
    column_read = ['start_district_id', 'dest_district_id', 'date', 'time', 'mean_his_day',
                   'mean_his_week', 'lagging_3', 'lagging_2', 'lagging_1', 'count']
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv', usecols=column_read)
    df.set_index(keys=['date', 'time', 'start_district_id', 'dest_district_id'], inplace=True)
    # df = df.sort_values(by=['start_district_id', 'dest_district_id', 'date', 'time'], axis=0, ascending=True)
    # df.to_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair.csv')

    # y_predict = list(map(lambda x: round(x), y_predict))

    mse = mean_squared_error(df['mean_his_day'], df['count'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(df['mean_his_day'], df['count'])
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(df['mean_his_day'], df['count'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(df['mean_his_day'], df['count'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    mse = mean_squared_error(df['mean_his_week'], df['count'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(df['mean_his_week'], df['count'])
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(df['mean_his_week'], df['count'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(df['mean_his_week'], df['count'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    mse = mean_squared_error(df['lagging_3'], df['count'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(df['lagging_3'], df['count'])
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(df['lagging_3'], df['count'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(df['lagging_3'], df['count'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    mse = mean_squared_error(df['lagging_2'], df['count'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(df['lagging_2'], df['count'])
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(df['lagging_2'], df['count'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(df['lagging_2'], df['count'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    mse = mean_squared_error(df['lagging_1'], df['count'])
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(df['lagging_1'], df['count'])
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(df['lagging_1'], df['count'])
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(df['lagging_1'], df['count'])
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    get_mean_his()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# mean_his_day
# MSE: 111.8363
# MAE: 3.7116
# MSLE: 0.2877
# r^2 on test data : 0.909675

# mean_his_week
# MSE: 105.2999
# MAE: 3.8360
# MSLE: 0.3573
# r^2 on test data : 0.917447

# lagging_3
# MSE: 463.2322
# MAE: 6.8378
# MSLE: 0.6676
# r^2 on test data : 0.716112

# lagging_2
# MSE: 288.8441
# MAE: 5.6852
# MSLE: 0.5506
# r^2 on test data : 0.822909

# lagging_1
# MSE: 123.2683
# MAE: 4.3018
# MSLE: 0.4461
# r^2 on test data : 0.924387

