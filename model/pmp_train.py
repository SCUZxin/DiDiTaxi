# proportion matrix based prediction model 基于比例矩阵的预测模型
# 使用梯度下降法求导进行迭代求参数
# 使用前 2 周的数据，预测第三周的数据

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


# 获取前三周的数据，并按照['date', 'time', 'start_district_id', 'dest_district_id']排序
# 新建一列 counter（1-1152）由 day + time 组合而成
# 包含所有数据的总流量1152，比例矩阵1152个，每个都是59*59，特征向量->特征相似度（实时计算）
def get_all_data():
    try:
        df_data = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\data_3_week.csv')
    except FileNotFoundError:
        # 读取Train 和 Test 进行拼接，按照 date、time、start_district_id、dest_district_id 进行排序，并加入counter
        df_train = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Train.csv')
        df_test = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv')
        df_data = pd.concat([df_train, df_test.loc[0:99123]], ignore_index=True)\
            .sort_values(by=['date', 'time', 'start_district_id', 'dest_district_id'], axis=0, ascending=True)\
            .reset_index(drop=True)
        df_data['counter'] = (df_data['day']-1) * 48 + df_data['time'] + 1
        df_data.set_index(keys=['date', 'time', 'start_district_id', 'dest_district_id'], inplace=True)
        df_data.to_csv('E:\\data\\DiDiData\\data_csv\\dataset\\data_3_week.csv')

    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\before\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    del df_time['date']
    del df_time['time']
    df_feature = pd.concat([df_weather, df_time], axis=1)
    df_feature['counter'] = range(1, 1153)
    flow_total = list(df_feature['count'].values)

    df_test = df_data.loc[319127:].reset_index(drop=True)
    flow_total_m = []
    for i in range(len(df_test)):
        flow_total_m.append(df_feature.loc[df_test.loc[i, 'counter']-1, 'count'])
    proportion_list = gen_proportion.get_proportion()

    return df_data, df_test, df_feature, flow_total, flow_total_m, proportion_list,


# 每一个预测OD对的参考历史的比例pi,顺序是weight_list_dict中key对应的list的顺序,key是要预测的OD对的顺序（1-m）
def get_Pti_his_t_dict():
    global df_data, df_test, flow_total, weight_list_dict, Pti_his_t_dict
    try:
        Pti_his_t_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\result\\params\\Pti_his_t_dict.pkl', 'rb'))
    except FileNotFoundError:
        print(len(df_test))
        print('m:', m)  # 165409

        for i in range(len(df_test)):
            s = df_test.loc[i, 'start_district_id']
            d = df_test.loc[i, 'dest_district_id']
            t = df_test.loc[i, 'counter']       # 要预测的OD对的counter=t
            weight_list_t = weight_list_dict[str(t)]    # 该counter=t需要参考的时间片list
            Pti_his_ti_list = []
            for ti in weight_list_t:  # 参考历史的counter = ti
                # 获取该counter=ti的总流量total
                flow_total_ti = flow_total[ti - 1]
                # 获取参考 counter=ti 时的OD对的流量
                count_df = df_data.loc[(df_data['start_district_id'] == s) & (df_data['dest_district_id'] == d)
                                       & (df_data['counter'] == ti)].reset_index(drop=True)
                if len(count_df) == 1:
                    count_ti = count_df.loc[0, 'count']
                else:
                    count_ti = 0
                Pti_his_ti_list.append(count_ti / flow_total_ti)
                # print('count_ti', count_ti)
                # print('weight_list_t', weight_list_t)
                # print('ti', ti)
                # print('count_ti', count_ti)
                # print('flow_total_ti', flow_total_ti)
                # print('Pti_his_ti_list', Pti_his_ti_list)

            Pti_his_t_dict[str(i+1)] = Pti_his_ti_list
        pickle.dump(Pti_his_t_dict, open('E:\\data\\DiDiData\\data_csv\\result\\params\\Pti_his_t_dict.pkl', 'wb'))
    # return Pti_his_t_dict


# 进行迭代求取最佳参数
def iteration(init_parameters, alpha=0.1, len_pre_t=3):
    global df_data, y_test, start_list, dest_list, Pti, flow_total
    list_t = [i for i in range(673, 1009)]
    # 下面这些值的list都是在整个迭代过程中不会改变的
    for i in range(len(list_t)):
        df_tmp = df_data.loc[df_data['counter'] == list_t[i]].reset_index(drop=True)
        y_test.extend(df_tmp['count'].values)
        Pti.extend(df_tmp['count'].values/flow_total[list_t[i]-1])
        start_list.extend(df_tmp['start_district_id'].values)
        dest_list.extend(df_tmp['dest_district_id'].values)

    with open('E:\\data\\DiDiData\\data_csv\\result\\rho_1_iterator.txt', 'a') as f:
        f.write('步长: ' + str(alpha) + '\n')
    i = 1
    last_loss = 0
    loss = 100
    last_rho_1 = init_parameters[0]
    new_rho_1 = 0
    last_rho_2 = init_parameters[1]
    new_rho_2 = 0
    while i < 1000:
        if loss == last_loss and last_rho_1 == new_rho_1 and last_rho_2 == new_rho_2:
            break
        print('iterator ', i)
        last_loss = loss
        loss = predict_batch(list_t, len_pre_t=len_pre_t, period=True,
                             rho_1=init_parameters[0], rho_2=init_parameters[1],
                             alpha1=init_parameters[2], alpha2=init_parameters[3])
        with open('E:\\data\\DiDiData\\data_csv\\result\\rho_1_iterator.txt', 'a') as f:
            f.write(str(init_parameters[0]) + '      ' + str(init_parameters[1])
                    + '       ' + str(init_parameters[2]) + '        ' + str(init_parameters[3])
                    + '       '+str(loss)+'\n')
        last_rho_1 = init_parameters[0]
        last_rho_2 = init_parameters[1]
        new_rho_1 = iterator_rho_1()
        new_rho_2 = iterator_rho_2()
        init_parameters[0] = new_rho_1
        init_parameters[1] = new_rho_2
        print('loss: ', loss, '  rho_1: ', init_parameters[0], '  rho_2: ', init_parameters[1])
        i += 1


# 计算使用上一个参数得到的每一个OD对的差值 error_m_list
def get_error_list():
    error_m_list = list(map(lambda x: x[0] - x[1], zip(y_predict, y_test)))
    return error_m_list


def iterator_rho_1():
    global m, alpha, y_test, df_test, init_parameters, flow_total_m
    error_m_list = get_error_list()
    sum1 = 0
    for i in range(1, m+1):
        # sum1 += (h_theta_ti_list[i] - Pti[i]) * (y_test[i]**2)\
        #         * derivative_h_theta_ti_to_rho_1(i, df_test.loc[i-1, 'counter'])
        sum1 += error_m_list[i-1] * flow_total_m[i-1] * derivative_h_theta_ti_to_rho_1(i, df_test.loc[i-1, 'counter'])

    # sum1 = sum1 * 1e14
    last_rho_1 = init_parameters[0]
    print('sum1', sum1)
    print('sum1/m', sum1/m)
    # rho_1=0.95,  sum1: 2.26953837628e-14      sum1/m: 1.37207671667e-19
    # rho_1=0.1,   sum1: 4.99027819526e-14      sum1/m: 3.01693269124e-19
    new_rho_1 = last_rho_1 - alpha*sum1/m
    return new_rho_1



def iterator_rho_2():
    global m, alpha, y_test, df_test, init_parameters, flow_total_m
    error_m_list = get_error_list()
    sum1 = 0
    for i in range(1, m+1):
        # sum1 += (h_theta_ti_list[i] - Pti[i]) * (y_test[i]**2)\
        #         * derivative_h_theta_ti_to_rho_1(i, df_test.loc[i-1, 'counter'])
        sum1 += error_m_list[i-1] * flow_total_m[i-1] * derivative_h_theta_ti_to_rho_2(i, df_test.loc[i-1, 'counter'])

    # sum1 = sum1 * 1e14
    last_rho_2 = init_parameters[1]
    print('rho2  sum1', sum1)
    print('rho2  sum1/m', sum1/m)
    # rho_1=0.95,  sum1: 2.26953837628e-14      sum1/m: 1.37207671667e-19
    # rho_1=0.1,   sum1: 4.99027819526e-14      sum1/m: 3.01693269124e-19
    new_rho_2 = last_rho_2 - alpha*sum1/m
    return new_rho_2


# hθ(ti) 对 ρ1 求偏导, counter=t 时的求导
def derivative_h_theta_ti_to_rho_1(i, t):
    global Pti, Pti_his_t_dict
    Pti_his_t_list = Pti_his_t_dict[str(i)]
    denominator = sum_of_weight_dict[str(t)]**2    # 分母
    weight_list = weight_list_dict[str(t)]  # 对counter = t，需要哪些时间片进行加权
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for j in range(len(weight_list)):
        omega_to_rho_1 = derivative_omega_to_rho(weight_list[j], t)[0]
        sum1 += omega_to_rho_1 * Pti_his_t_list[j]   #################################################################
        sum2 += omega_to_rho_1
        sum3 += weight_dict[str(weight_list[j])+'-'+str(t)] * Pti_his_t_list[j]
    #     print('weight_list', weight_list)
    #     print('omega_to_rho_1', omega_to_rho_1)
    #     print('Pti_his_t_list[',j,']', Pti_his_t_list[j])
    #     print('weight_dict[',str(weight_list[j])+'-'+str(t),']',weight_dict[str(weight_list[j])+'-'+str(t)])
    #
    # print('Pti_his_t_dict[',i,']', Pti_his_t_dict[str(i)])
    # print('sum1 :', sum1)
    # print('sum_of_weight_dict[str(t)] :', sum_of_weight_dict[str(t)])
    # print('sum2 :', sum2)
    # print('sum3 :', sum3)
    numerator = (sum1 * sum_of_weight_dict[str(t)]) - (sum2 * sum3)   # 分子
    # print('numerator :', numerator)
    return numerator / denominator


# hθ(ti) 对 ρ2 求偏导, counter=t 时的求导
def derivative_h_theta_ti_to_rho_2(i, t):
    global Pti, Pti_his_t_dict
    Pti_his_t_list = Pti_his_t_dict[str(i)]
    denominator = sum_of_weight_dict[str(t)]**2    # 分母
    weight_list = weight_list_dict[str(t)]  # 对counter = t，需要哪些时间片进行加权
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for j in range(len(weight_list)):
        omega_to_rho_2 = derivative_omega_to_rho(weight_list[j], t)[1]
        sum1 += omega_to_rho_2 * Pti_his_t_list[j]   #################################################################
        sum2 += omega_to_rho_2
        sum3 += weight_dict[str(weight_list[j])+'-'+str(t)] * Pti_his_t_list[j]
    #     print('weight_list', weight_list)
    #     print('omega_to_rho_1', omega_to_rho_1)
    #     print('Pti_his_t_list[',j,']', Pti_his_t_list[j])
    #     print('weight_dict[',str(weight_list[j])+'-'+str(t),']',weight_dict[str(weight_list[j])+'-'+str(t)])
    #
    # print('Pti_his_t_dict[',i,']', Pti_his_t_dict[str(i)])
    # print('sum1 :', sum1)
    # print('sum_of_weight_dict[str(t)] :', sum_of_weight_dict[str(t)])
    # print('sum2 :', sum2)
    # print('sum3 :', sum3)
    numerator = (sum1 * sum_of_weight_dict[str(t)]) - (sum2 * sum3)   # 分子
    # print('numerator :', numerator)
    return numerator / denominator


def derivative_omega_to_rho(t1, t):
    global df_feature, init_parameters
    rho_1 = init_parameters[0]
    rho_2 = init_parameters[1]
    lambda_wtr = lambda_weather(t1, t)
    one_t1_t2, period = lambda_is_voca_week(t1, t)
    if one_t1_t2 == 0:
        return [0, 0]
    d_t1_t2 = abs(t1 - t) % 48
    delta_h = min(d_t1_t2, 48-d_t1_t2)
    # print(delta_h)
    # delta_d = math.ceil(abs(t1-t2)/48)
    # 如果是closeness, delta_d = 0， 如果是 period， delta_d = abs(day2 - day1)
    delta_d = 0
    if period:
        delta_d = abs(df_feature.loc[t-1, 'day'] - df_feature.loc[t1-1, 'day'])
    omega_to_rho_1 = delta_h * pow(rho_1, delta_h-1) * pow(rho_2, delta_d) * lambda_wtr
    omega_to_rho_2 = pow(rho_1, delta_h) * delta_d * pow(rho_2, delta_d - 1) * lambda_wtr
    return [omega_to_rho_1, omega_to_rho_2]



def derivative_alpha_1():
    pass


def derivative_alpha_2():
    pass



def get_general_item():
    pass


# 单个时间片的预测，第 t (1-1152) 个时间片的订单量，需要前面 1-H ， 此处先用3个时间片
def predict_single(t=817, len_pre_t=3, period=False):
    global flow_total, weight_list_dict, weight_dict
    # 需要的前 1-H 个时间片 closeness
    closeness_time_list = [i for i in range(t-len_pre_t, t)]

    # 上几周该时间片 period
    period_time_list = []
    ind = t
    while ind > 7*48:
        ind = ind - 7*48
        period_time_list.append(ind)

    w_closeness = []
    prop_closeness = []
    sum_prop_closeness = np.zeros((59, 59))
    # closeness 计算这些时间片的特征相似性
    for t1 in closeness_time_list:
        weight = w_features(t1, t)
        weight_dict[str(t1) + '-' + str(t)] = weight
        w_closeness.append(weight)
        prop_closeness.append(proportion_list[t1-1])

    # period 计算这些时间片的特征相似性
    w_period = []
    prop_period = []
    sum_prop_period = np.zeros((59, 59))
    if period:
        for t1 in period_time_list:
            weight = w_features(t1, t)
            weight_dict[str(t1)+'-'+str(t)] = weight
            w_period.append(weight)
            prop_period.append(proportion_list[t1 - 1])

    # 加权平均
    # prop_t = sum_prop_closeness / sum_w_closeness
    # print('w_closeness:', w_closeness)
    # print('w_period:', w_period)
    for i in range(len(w_closeness)):
        sum_prop_closeness = sum_prop_closeness + (w_closeness[i]*prop_closeness[i])
    for i in range(len(w_period)):
        sum_prop_period = sum_prop_period + (w_period[i]*prop_period[i])

    # 预测counter=t的流量矩阵需要哪几个时间片加权，key是counter=t
    weight_list_dict[str(t)] = period_time_list+closeness_time_list
    sum_of_weight = np.sum(w_closeness)+np.sum(w_period)
    prop_t = (sum_prop_closeness+sum_prop_period) / sum_of_weight
    # 流量矩阵 = 总流量 * prop_t，得到59*59的数组
    flow_pair_t = flow_total[t-1] * prop_t

    # print('prop_t', prop_t[1][42])
    # print('prop_t', prop_t[2][2])
    # print('flow_pair_t', flow_pair_t[1][42])
    # print('flow_pair_t', flow_pair_t[2][2])
    return flow_pair_t, sum_of_weight, prop_t


# 批量预测，预测多个时间片的流量
def predict_batch(list_t, len_pre_t=3, period=False, rho_1=0.9, rho_2=0.9, alpha1=4, alpha2=10):
    # with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
    #     f.write('predict_batch(['+str(min(list_t))+'……'+str(max(list_t))+'], len_pre_t='+str(len_pre_t)
    #             +', period='+str(period)+', rho_1='+str(rho_1)+', rho_2='+str(rho_2)
    #             +', alpha_1='+str(alpha1)+', alpha_2='+str(alpha2)+') :\n')
    global df_data, flow_pair_batch_list, sum_of_weight_dict, prop_t_list, h_theta_ti_list
    # 每一轮迭代就用不同的参数，所以要初始化重新更新
    flow_pair_batch_list = []   # 每一个元素是一个预测时间片的流量矩阵
    sum_of_weight_dict = {}     # 每一个元素是一个预测时间片的权值之和
    prop_t_list = []     # 每一个元素是一个预测时间片的比例矩阵
    # h_theta_ti_list = []    # 所有OD对的预测比例的list
    for t in list_t:
        flow_pair_t, sum_of_weight, prop_t = predict_single(t, len_pre_t, period)
        sum_of_weight_dict[str(t)] = sum_of_weight
        prop_t_list.append(prop_t)
        flow_pair_batch_list.append(flow_pair_t)

        # df_tmp = df_data.loc[df_data['counter'] == t].reset_index(drop=True)
        # for i in range(len(df)):
        #     h_theta_ti_list.append(prop_t[df_tmp.loc[i, 'start_district_id'][df_tmp.loc[i, 'dest_district_id']])
    # print('xxxxxxxxxxxxxxx')
    # print(flow_pair_batch_list[0][1][42])
    # print(flow_pair_batch_list[0][2][2])
    if len(Pti_his_t_dict) == 0:
        get_Pti_his_t_dict()
    return metrics_self(flow_pair_batch_list, list_t)


# flow_pair_batch_list：按照时间 list_t 装多个数组，每个list是对应时间片的订单量的数组 59*59
def metrics_self(flow_pair_batch_list, list_t):
    global df_data, y_predict, y_test, init_parameters
    # 每次重新计算需要重置 y_predict
    y_predict = []
    for i in range(len(list_t)):
        df_tmp = df_data.loc[df_data['counter'] == list_t[i]].reset_index(drop=True)
        for j in range(len(df_tmp)):
            s = df_tmp.loc[j, 'start_district_id']
            d = df_tmp.loc[j, 'dest_district_id']
            # print(s)
            # print(d)
            # print(flow_pair_batch_list[i][s][d])
            y_predict.append(flow_pair_batch_list[i][s][d])

    # print('y_predict', y_predict[0:100])

    # for i in range(len(y_predict)):
    #     y_predict[i] = round(y_predict[i])

    # 平方误差函数，最小二乘法构建损失函数
    loss = mean_squared_error(y_test, y_predict) / 2

    return loss


def w_features(t1, t2):
    w_t1_t2 = lambda_time(t1, t2) * lambda_weather(t1, t2)
    # print('t1',t1)
    # print('t2',t2)
    # print('lambda_time(t1, t2)',lambda_time(t1, t2))
    return w_t1_t2


def lambda_is_voca_week(t1, t2):
    global df_feature
    period = True
    # closeness,，时间片很接近
    if abs(t2 - t1) <= 20:
        period = False
    # 只要不是同是工作日/周末，不同是假日/非假日，lambda_1 = 0
    # 这个只对period生效，closeness 如何衡量
    if period:
        if df_feature.loc[t1-1, 'is_vocation'] != df_feature.loc[t2-1, 'is_vocation']:
            return 0, period
        elif df_feature.loc[t1-1, 'is_weekend'] != df_feature.loc[t2-1, 'is_weekend']:
            return 0, period
    return 1, period


# 计算时间的相似性, t: 1-1152
def lambda_time(t1, t2):
    global df_feature, init_parameters
    rho_1 = init_parameters[0]
    rho_2 = init_parameters[1]
    one_t1_t2, period = lambda_is_voca_week(t1, t2)
    if one_t1_t2 == 0:
        return 0
    d_t1_t2 = abs(t1 - t2) % 48
    delta_h = min(d_t1_t2, 48-d_t1_t2)
    # print(delta_h)
    # delta_d = math.ceil(abs(t1-t2)/48)
    # 如果是closeness, delta_d = 0， 如果是 period， delta_d = abs(day2 - day1)
    delta_d = 0
    if period:
        delta_d = abs(df_feature.loc[t2-1, 'day'] - df_feature.loc[t1-1, 'day'])
    lambda_1 = pow(rho_1, delta_h) * pow(rho_2, delta_d)
    return lambda_1


# 计算天气的相似性 t: 1-1152
def lambda_weather(t1, t2):
    global df_feature, max_temperature, min_temperature, max_pm25, min_pm25, init_parameters
    alpha1 = init_parameters[2]
    alpha2 = init_parameters[3]
    # 是用时间片的 weather 还是当天的 weather
    weather_t1 = df_feature.loc[t1-1, 'weather']
    weather_t2 = df_feature.loc[t2-1, 'weather']
    # alpha_1……6
    if weather_t1 == weather_t2:
        lambda_2 = 1
    elif weather_t1 == 4:
        lambda_2 = weather_t2 / alpha2
    elif weather_t2 == 4:
        lambda_2 = weather_t1 / alpha2
    else:
        lambda_2 = 1 - abs(weather_t2 - weather_t1) / alpha1

    lambda_3 = 1 - (abs(df_feature.loc[t1-1, 'temperature'] - df_feature.loc[t2-1, 'temperature']) / (max_temperature - min_temperature))
    lambda_4 = 1 - (abs(df_feature.loc[t1-1, 'pm2.5'] - df_feature.loc[t2-1, 'pm2.5']) / (max_pm25 - min_pm25))

    # print('t1',t1)
    # print('t2',t2)
    # print('weather_t1_t2', lambda_2*lambda_3*lambda_4)

    return lambda_2*lambda_3*lambda_4


# 使用2-D 高斯核函数 计算天气的相似性 t: 1-1152
def lambda_weather_2d(t1, t2, alpha1=4, alpha2=10, sigma1=1, sigma2=1):
    global df_feature, max_temperature, min_temperature, max_pm25, min_pm25
    # 是用时间片的 weather 还是当天的 weather
    weather_t1 = df_feature.loc[t1-1, 'weather']
    weather_t2 = df_feature.loc[t2-1, 'weather']
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

    temp_1 = df_feature.loc[t1-1, 'temperature']
    temp_2 = df_feature.loc[t2-1, 'temperature']
    pm25_1 = df_feature.loc[t1-1, 'pm2.5']
    pm25_2 = df_feature.loc[t2-1, 'pm2.5']
    k_t1_t2 = math.exp(-(((temp_1-temp_2)**2)/(sigma1**2)
                         + ((pm25_1-pm25_2)**2)/(sigma2**2))) / (2*math.pi*sigma1*sigma2)

    return lambda_2 * k_t1_t2


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # flow_total_m: 1-m 个预测OD对所在时间片上的城市总流量
    df_data, df_test, df_feature, flow_total, flow_total_m, proportion_list = get_all_data()
    max_temperature = df_data['temperature'].max()
    min_temperature = df_data['temperature'].min()
    max_pm25 = df_data['pm2.5'].max()
    min_pm25 = df_data['pm2.5'].min()

    m = 165409  # 一共有这么多个OD对需要预测
    y_predict = []  # 每一轮迭代的预测值
    y_test = []
    start_list, dest_list = [], []  # 按照顺序排序得到的起始地、目的地的list
    Pti = []    # 每一个要预测的OD对的实际比例，一共有m个
    # 每一个预测OD对的参考历史的比例pi,顺序是weight_list_dict中key对应的list的顺序, key是要预测的OD对的顺序（1-m）
    Pti_his_t_dict = {}
    # 每一轮迭代的流量矩阵 、 权值（特征相似度）之和(key是counter=t)、 比例矩阵的list
    flow_pair_batch_list, sum_of_weight_dict, prop_t_list = [], {}, []
    h_theta_ti_list = []  # 所有OD对的预测比例的list
    weight_list_dict = {}   # 对每一个 counter = t 进行计算时需要加权的时间片list, key是counter=t
    weight_dict = {}     # 任意两个时间片（counter=t1,t2）的权值
    # 初始化所有参数 rho_1, rho_2, alpha_1, alpha_2
    # init_parameters = [1, 1, 10, 10]
    init_parameters = [0.9, 0.95, 4, 10]
    alpha = 0.01
    len_pre_t = 3
    iteration(init_parameters, alpha, len_pre_t)

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# 结论：
# derivative_h_theta_ti_to_rho_1,
# 求导出来的结果很小（即：分子相减的结果），导致参数没有变化，应该是不收敛？？？




