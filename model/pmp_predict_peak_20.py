# 只预测早晚峰值(7:30-9:30)[15,18]和(16:30-19:00)[33, 37]，用20% 中聚类为1,2,3的 OD 对

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from datetime import datetime

def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i]-y_predict[i]) / y_real[i]
    return sum1/len(y_predict)


def get_od_three_cluster_dict():
    # 读取 time_seg_cluster_result.csv 文件找到 20% 的OD对
    df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result.csv')
    od_dict = {}
    for i in range(len(df_cluster)):
        s = df_cluster.loc[i, 'start_district_id']
        d = df_cluster.loc[i, 'dest_district_id']
        if df_cluster.loc[i, 'real_num'] != 4:
            od_dict[(str(s) + '-' + str(d))] = df_cluster.loc[i, 'real_num']
    return od_dict


# 3类OD对全部用于早晚高峰
def peak_predict():
    global od_three_cluster_dict
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair_result.csv')
    df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_result.csv')

    # 结果从 pmp_pair_predict_final.csv 的预测结果文件 pmp_pair_result.csv 获取
    # 只取 OD 对在20%的那三类，且时间片符合要求的
    time_match = [15, 16, 17, 18, 33, 34, 35, 36, 37]
    df_match = pd.DataFrame()
    for i in range(len(df_pmp_result)):
        if i % 10000 == 0:
            print('iterator: ', i)
        s = df_pmp_result.loc[i, 'start_district_id']
        d = df_pmp_result.loc[i, 'dest_district_id']
        time = df_pmp_result.loc[i, 'time']
        key = str(s)+'-'+str(d)
        if key in od_three_cluster_dict and time in time_match:
            df_match = df_match.append(df_pmp_result.loc[i])

    df_match = df_match.reset_index(drop=True)
    df_match.to_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_peak_result.csv')
    y_predict = list(df_match['predict_count'].values)
    y_test = list(df_match['real_count'].values)
    mse = mean_squared_error(y_predict, y_test)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_predict, y_test)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    mape = mean_absolute_perc_error(y_predict, y_test)
    print("MAPE: %.4f" % mape)  # 输出平均百分比绝对误差
    me = max(list(map(lambda x1,x2:abs(x1-x2), y_predict,y_test)))
    print("ME: %.4f" % me)  # 输出最大误差
    msle = mean_squared_log_error(y_predict, y_test)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_predict, y_test)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


# 簇1,3：早高峰； 簇2,3：晚高峰
def early_late_peak_predict():
    global od_three_cluster_dict
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair.csv')
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_result_4_20_GBRT.csv')
    df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\KNN\\KNN_result_3_6.csv')
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_result_NMF_AR.csv')
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmwa_pair_result.csv')

    # 结果从 pmp_pair_predict_final.csv 的预测结果文件 pmp_pair_result.csv 获取
    # 只取 OD 对在20%的那三类，且时间片符合要求的
    early_peak = [15, 16, 17, 18]
    late_peak = [33, 34, 35, 36, 37]
    df_match = pd.DataFrame()
    for i in range(len(df_pmp_result)):
        if i % 10000 == 0:
            print('iterator: ', i)
        s = df_pmp_result.loc[i, 'start_district_id']
        d = df_pmp_result.loc[i, 'dest_district_id']
        time = df_pmp_result.loc[i, 'time']
        key = str(s)+'-'+str(d)
        # 只添加早高峰
        if key in od_three_cluster_dict:
            value = od_three_cluster_dict[key]
            if (value == 1 or value == 3) and time in early_peak:
                df_match = df_match.append(df_pmp_result.loc[i])
            if (value == 2 or value == 3) and time in late_peak:
                df_match = df_match.append(df_pmp_result.loc[i])

    df_match = df_match.reset_index(drop=True)
    df_match.to_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_peak_result.csv')
    y_predict = list(df_match['predict_count'].values)
    y_test = list(df_match['real_count'].values)
    mse = mean_squared_error(y_predict, y_test)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_predict, y_test)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    mape = mean_absolute_perc_error(y_predict, y_test)
    print("MAPE: %.4f" % mape)  # 输出平均百分比绝对误差
    me = max(list(map(lambda x1,x2:abs(x1-x2), y_predict,y_test)))
    print("ME: %.4f" % me)  # 输出最大误差
    msle = mean_squared_log_error(y_predict, y_test)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_predict, y_test)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    od_three_cluster_dict = get_od_three_cluster_dict()
    # gbrt_pair_result.csv 需要改为 pmp_pair_result.csv
    # peak_predict()
    early_late_peak_predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 使用 HA 的结果:
# MSE: 567.4351
# MAE: 11.5550
# MAPE: 0.3709
# ME: 538.0000
# MSLE: 0.2077
# r^2 on test data : 0.896886

# GBRT:
# MSE: 571.3905
# MAE: 11.7757
# MAPE: 0.3966
# ME: 365.0000
# MSLE: 0.1799
# r^2 on test data : 0.914063

# KNN:
# MSE: 348.9112
# MAE: 9.5885
# MAPE: 0.3902
# ME: 273.0000
# MSLE: 0.1906
# r^2 on test data : 0.947390


# NMF-AR
# MSE: 414.4990
# MAE: 10.5865
# MAPE: 0.4138
# ME: 236.0000
# MSLE: 0.1987
# r^2 on test data : 0.934631

# PMWA
# MSE: 318.4044
# MAE: 9.2075
# MAPE: 0.3694
# ME: 237.0000
# MSLE: 0.1527
# r^2 on test data : 0.950223