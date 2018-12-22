# 只预测早晚峰值(7:30-9:30)[15,18]和(16:30-19:00)[33, 37]，用20% 中聚类为1,2,3的 OD 对

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from datetime import datetime


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
    msle = mean_squared_log_error(y_predict, y_test)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_predict, y_test)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    od_three_cluster_dict = get_od_three_cluster_dict()
    # gbrt_pair_result.csv 需要改为 pmp_pair_result.csv
    peak_predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 使用 ha_pair_result.csv 的结果:
# MSE: 446.1751
# MAE: 9.4219
# MSLE: 0.1871
# r^2 on test data : 0.890024

# 使用 gbrt_pair_result.csv 的结果:
# MSE: 296.9339
# MAE: 8.1962
# MSLE: 0.1717
# r^2 on test data : 0.935794

# 使用 pmp_pair_result.csv 的结果:
# MSE: 270.1761
# MAE: 8.1990
# MSLE: 0.2053
# r^2 on test data : 0.944540

