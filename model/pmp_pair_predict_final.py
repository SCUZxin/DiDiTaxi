# proportion matrix based prediction model 基于比例矩阵的预测模型
# 先不用全部预测，只是简单预测几个时间片试一下效果

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle

import extract_feature.gen_proportion as gen_proportion
import extract_feature.traffic_similarity as tc_similarty


# 获取所有训练数据和测试数据，新建一列 counter（1-1152）由 day + time 组合而成
# 包含所有数据的总流量1152，比例矩阵1152个，每个都是59*59，特征向量->特征相似度（实时计算）
def get_all_data():
    # 读取测试集并加入counter
    df_test = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv')
    df_test['counter'] = (df_test['day']-1) * 48 + df_test['time'] + 1

    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\before\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    del df_time['date']
    del df_time['time']
    df = pd.concat([df_weather, df_time], axis=1)
    df['counter'] = range(1, 1153)
    # flow_total = list(df['count'].values)

    df_tatol_predict = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\gbrt_toal_result.csv')
    flow_total = list(df_tatol_predict['count'].values)

    proportion_list = gen_proportion.get_proportion()
    return df, flow_total, proportion_list, df_test


# 单个时间片的预测，第 t (1-1152) 个时间片的订单量，需要前面 1-H ， 此处先用3个时间片
def predict_single(t=817, len_pre_t=3):
    global flow_total, period
    # 需要的前 1-H 个时间片 closeness
    closeness_time_list = [i for i in range(t-len_pre_t, t)]

    # 上几周该时间片 period
    period_time_list = []
    ind = t
    while ind > 7*48:
        ind = ind - 7*48
        period_time_list.append(ind)

    w_array_closeness = []
    prop_closeness = []     # 对应时间片的比例矩阵
    sum_prop_closeness = np.zeros((58, 58))
    # closeness 计算这些时间片的特征相似性
    for t1 in closeness_time_list:
        weight_array = w_features(t1, t)
        w_array_closeness.append(weight_array)
        prop_closeness.append(proportion_list[t1 - 1])
        # w_closeness.append(w_features(t1, t))
        # prop_closeness.append(proportion_list[t1-1])

    # period 计算这些时间片的特征相似性
    w_array_period = []
    prop_period = []
    sum_prop_period = np.zeros((58, 58))
    if period:
        for t1 in period_time_list:
            weight_array = w_features(t1, t)
            w_array_period.append(weight_array)
            prop_period.append(proportion_list[t1 - 1])

    # 加权平均
    # prop_t = sum_prop_closeness / sum_w_closeness
    sum_of_weight = np.zeros((58, 58))
    for i in range(len(w_array_closeness)):
        sum_of_weight = sum_of_weight + w_array_closeness[i]
        sum_prop_closeness = sum_prop_closeness + (w_array_closeness[i]*np.array(prop_closeness[i]))
    for i in range(len(w_array_period)):
        sum_of_weight = sum_of_weight + w_array_period[i]
        sum_prop_period = sum_prop_period + (w_array_period[i]*np.array(prop_period[i]))

    prop_t = (sum_prop_closeness + sum_prop_period) / sum_of_weight
    # 流量矩阵 = 总流量 * prop_t，得到58*58的数组
    prop_t[np.isnan(prop_t)] = 1
    prop_t[np.isinf(prop_t)] = 1
    flow_pair_t = flow_total[t-817] * prop_t
    return flow_pair_t


# 批量预测，预测多个时间片的流量
def predict_batch(list_t, len_pre_t=3):
    global df_test, init_parameters, traffic_similarity_dict, period, df_target
    rho_1 = init_parameters[0]
    rho_2 = init_parameters[1]
    alpha1 = init_parameters[2]
    alpha2 = init_parameters[3]
    with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
        # f.write('predict_batch(['+','.join(str(i) for i in list_t)+'], len_pre_t='+str(len_pre_t)+', period='+str(period)+') :\n')
        # f.write('predict_batch(['+str(list_t[0])+'……'+str(list_t[len(list_t)-1])+'], len_pre_t='+str(len_pre_t)+', period='+str(period)+') :\n')
        f.write('predict_batch(['+str(min(list_t))+'……'+str(max(list_t))+'], len_pre_t='+str(len_pre_t)
                +', period='+str(period)+', rho_1='+str(rho_1)+', rho_2='+str(rho_2)
                +', alpha_1='+str(alpha1)+', alpha_2='+str(alpha2)+') :\n')

    if len(traffic_similarity_dict) == 0:
        traffic_similarity_dict = tc_similarty.get_traffic_similarity_dict()

    flow_pair_batch_list = []
    for t in list_t:
        flow_pair_batch_list.append(predict_single(t, len_pre_t))

    if len(df_target) == 0:
        for i in range(len(list_t)):
            df_tmp = df_test.loc[df_test['counter'] == list_t[i]]
            df_target = pd.concat([df_target, df_tmp]).reset_index(drop=True)

    metrics_self(flow_pair_batch_list, df_target, list_t)


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i]-y_predict[i]) / y_real[i]
    return sum1/len(y_predict)

# flow_pair_batch_list：按照时间 list_t 装多个list，每个list是对应时间片的订单量的数组 59*59
def metrics_self(flow_pair_batch_list, df_target, list_t):
    y_test = []
    y_predict = []
    for i in range(len(list_t)):
        df_tmp = df_target.loc[df_target['counter'] == list_t[i]].reset_index(drop=True)
        y_test.extend(df_tmp['count'].values)
        for j in range(len(df_tmp)):
            s = df_tmp.loc[j, 'start_district_id']
            d = df_tmp.loc[j, 'dest_district_id']
            y_predict.append(flow_pair_batch_list[i][s-1][d-1])

    y_predict = list(map(lambda x: round(x), y_predict))
    save_result(y_predict, y_test)
    # 如果遇到负数转为正数，免得计算MSLE出错
    y_predict = list(map(lambda x: -x if x < 0 else x, y_predict))
    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    mape = mean_absolute_perc_error(y_predict, y_test)
    print("MAPE: %.4f" % mape)  # 输出平均百分比绝对误差
    me = max(list(map(lambda x1,x2:abs(x1-x2), y_predict,y_test)))
    print("ME: %.4f" % me)  # 输出最大误差
    msle = mean_squared_log_error(y_test, y_predict)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
        # f.write('  MSE          MAE         MLSE            r^2\n')
        f.write('%.4f' % mse+'      %.4f' % mae+'       %.4f' % msle+'      %f' % r2+'\n')


def w_features(t1, t2):
    global traffic_similarity_dict
    traffic_similarity_array = np.zeros((58, 58))
    for i in range(1, 59):
        traffic_similarity_array[i-1] = traffic_similarity_dict[str(i)+':'+str(t1)+'-'+str(t2)]
        traffic_similarity_array[np.isnan(traffic_similarity_array)] = 1
        traffic_similarity_array[np.isinf(traffic_similarity_array)] = 1

    w_t1_t2 = lambda_time(t1, t2) * lambda_weather(t1, t2) * traffic_similarity_array
    return w_t1_t2


# 计算时间的相似性, t: 1-1152
def lambda_time(t1, t2):
    global df, init_parameters
    rho1 = init_parameters[0]
    rho2 = init_parameters[1]
    period_1 = True
    # closeness,，时间片很接近
    if abs(t2 - t1) <= 20:
        period_1 = False
    # 只要不是同是工作日/周末，不同是假日/非假日，lambda_1 = 0
    # 这个只对period生效，closeness 如何衡量
    if period_1:
        if df.loc[t1-1, 'is_vocation'] != df.loc[t2-1, 'is_vocation']:
            return 0
        elif df.loc[t1-1, 'is_weekend'] != df.loc[t2-1, 'is_weekend']:
            return 0

    d_t1_t2 = abs(t1 - t2) % 48
    delta_h = min(d_t1_t2, 48-d_t1_t2)
    # print(delta_h)
    # delta_d = math.ceil(abs(t1-t2)/48)
    # 如果是closeness, delta_d = 0， 如果是 period， delta_d = abs(day2 - day1)
    delta_d = 0
    if period_1:
        delta_d = abs(df.loc[t2-1, 'day'] - df.loc[t1-1, 'day'])
    lambda_1 = pow(rho1, delta_h) * pow(rho2, delta_d)

    return lambda_1


# 计算天气的相似性 t: 1-1152
def lambda_weather(t1, t2):
    global df, max_temperature, min_temperature, max_pm25, min_pm25, init_parameters
    # 是用时间片的 weather 还是当天的 weather
    alpha1 = init_parameters[2]
    alpha2 = init_parameters[3]
    weather_t1 = df.loc[t1-1, 'weather']
    weather_t2 = df.loc[t2-1, 'weather']
    # alpha_1……6
    if weather_t1 == weather_t2:
        lambda_2 = 1
    elif weather_t1 == 4:
        lambda_2 = weather_t2 / alpha2
    elif weather_t2 == 4:
        lambda_2 = weather_t1 / alpha2
    else:
        lambda_2 = 1 - abs(weather_t2 - weather_t1) / alpha1

    lambda_3 = 1 - (abs(df.loc[t1-1, 'temperature'] - df.loc[t2-1, 'temperature']) / (max_temperature - min_temperature))
    lambda_4 = 1 - (abs(df.loc[t1-1, 'pm2.5'] - df.loc[t2-1, 'pm2.5']) / (max_pm25 - min_pm25))

    return lambda_2*lambda_3*lambda_4


# 使用2-D 高斯核函数 计算天气的相似性 t: 1-1152
def lambda_weather_2d(t1, t2, sigma1, sigma2):
    global df, max_temperature, min_temperature, max_pm25, min_pm25
    # 是用时间片的 weather 还是当天的 weather
    alpha1 = init_parameters[2]
    alpha2 = init_parameters[3]
    weather_t1 = df.loc[t1-1, 'weather']
    weather_t2 = df.loc[t2-1, 'weather']
    # alpha_1……6
    if weather_t1 == weather_t2:
        lambda_2 = 1
    elif weather_t1 == 4:
        lambda_2 = weather_t2 / alpha2
    elif weather_t2 == 4:
        lambda_2 = weather_t1 / alpha2
    else:
        lambda_2 = 1 - abs(weather_t2 - weather_t1) / alpha1

    # 使用 2-D Gaussian Kernal function 计算温度和pm2.5 的相似度

    temp_1 = df.loc[t1-1, 'temperature']
    temp_2 = df.loc[t2-1, 'temperature']
    pm25_1 = df.loc[t1-1, 'pm2.5']
    pm25_2 = df.loc[t2-1, 'pm2.5']
    k_t1_t2 = math.exp(-(((temp_1-temp_2)**2)/(sigma1**2)
                         + ((pm25_1-pm25_2)**2)/(sigma2**2))) / (2*math.pi*sigma1*sigma2)

    return lambda_2 * k_t1_t2


def save_result(y_predict, y_test):
    pass
    global df_test
    result = pd.DataFrame()
    # result['test_id'] = range(len(yp))
    result['date'] = df_test['date']
    result['time'] = df_test['time']
    result['start_district_id'] = df_test['start_district_id']
    result['dest_district_id'] = df_test['dest_district_id']
    result['predict_count'] = y_predict
    result['real_count'] = y_test
    result.to_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_result.csv', index=False)


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df, flow_total, proportion_list, df_test = get_all_data()
    max_temperature = df['temperature'].max()
    min_temperature = df['temperature'].min()
    max_pm25 = df['pm2.5'].max()
    min_pm25 = df['pm2.5'].min()
    traffic_similarity_dict = {}
    period = True
    df_target = pd.DataFrame()
    # init_parameters = [0.182428388207, 0.880798249348, 4.71840657989, 1.52477764084]
    init_parameters = [0.5565, 0.8916, 4.5038, 3.990159]
    # init_parameters = [0.34, 1.1, 6, 4]
    predict_batch([i for i in range(817, 1153)], len_pre_t=1)


    # # rho_1:0.8-0.99 间隔0.01
    # with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
    #     f.write('rho_1:\n')
    # for rho_1 in range(30, 50):
    #     init_parameters = [rho_1/100, 0.95, 4, 10]
    #     predict_batch([i for i in range(817, 1153)], len_pre_t=1)
    # with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
    #     f.write('\n\n\nrho_2: \n')
    #
    # # rho_2:0.8-0.99 间隔0.01
    # for rho_2 in range(90, 110):
    #     init_parameters = [0.8, rho_2/100, 4, 10]
    #     predict_batch([i for i in range(817, 1153)], len_pre_t=1)
    # with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
    #     f.write('\n\n\nalpha_1: \n')
    #
    # # alpha_1: 1-10
    # for alpha_1 in range(2, 10):
    #     init_parameters = [0.8, 0.95, alpha_1, 10]
    #     predict_batch([i for i in range(817, 1153)], len_pre_t=1)
    # with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
    #     f.write('\n\n\nalpha_2: \n')
    #
    # # alpha_2: 5-20
    # for alpha_2 in range(2, 10):
    #     init_parameters = [0.8, 0.95, 4, alpha_2]
    #     predict_batch([i for i in range(817, 1153)], len_pre_t=1)

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# 结论：

# 1. len_pre_t：取 1 要好些
# 2. rho_1：值越大效果越差，再取小一点的值试一下
# 3. rho_2：越大效果越好，再试试大于 1 的值， 大于 1 说明有递增趋势，用户增长因子？？？
# 4. alpha_1: 越大效果越好，为什么等于1要出问题 RuntimeWarning
# 5. alpha_2: 越小效果越好
# 6. 试试不用closeness， 只用 period 进行加权平均
# 7. 试试用四舍五入之后的结果进行评价
# 7. temperature 和 pm2.5 的相似度用二维高斯核函数试试
# 8. 用户增长因子如何确定？


# [0.5565, 0.8916, 4.5038, 3.990159]  pre_len=1
# MSE: 63.1680
# MAE: 3.3750
# MAPE: 0.6207
# ME: 301.0000
# MSLE: 0.3534
# r^2 on test data : 0.960986


