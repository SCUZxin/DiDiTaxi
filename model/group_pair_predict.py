# 根据OD对的流量范围进行分情况预测：80%使用HA, 20%使用PMP

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import pickle
from datetime import datetime


# 获取20% 和 80%的 OD对并 return
# od_20_dict：key(52-58), value: 属于哪个类（1-4）
def get_20_od_dict():
    try:
        od_20_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\od_20_dict.pkl', 'rb'))
    except FileNotFoundError:
        # 读取 time_seg_cluster_result.csv 文件找到 20% 的OD对
        df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result.csv')
        od_20_dict = {}
        for i in range(len(df_cluster)):
            s = df_cluster.loc[i, 'start_district_id']
            d = df_cluster.loc[i, 'dest_district_id']
            od_20_dict[(str(s)+'-'+str(d))] = df_cluster.loc[i, 'real_num']
        pickle.dump(od_20_dict, open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\od_20_dict.pkl', 'wb'))
    return od_20_dict


def group_predict():
    df_ha_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair_result.csv')
    # df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\gbrt_pair_result.csv')
    df_pmp_result = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_result.csv')

    # 80%的结果从 ha_pair_predict.csv 的预测结果文件 ha_pair.csv 获取, 没有出现过的OD对的流量都置为0
    # 20%的结果从 pmp_pair_predict_final.csv 的预测结果文件 pmp_pair_result.csv 获取，然后组合

    for i in range(len(df_ha_result)):
        if i % 10000 == 0:
            print('iterator: ', i)
        s = df_ha_result.loc[i, 'start_district_id']
        d = df_ha_result.loc[i, 'dest_district_id']
        key = str(s)+'-'+str(d)
        if key in od_20_dict:
            df_ha_result.loc[i, 'predict_count'] = df_pmp_result.loc[i, 'predict_count']

    print(df_ha_result.head())
    y_predict = list(df_ha_result['predict_count'].values)
    # y_predict = list(map(lambda x: round(x), y_predict))
    y_test = list(df_ha_result['real_count'].values)
    df_ha_result.to_csv('E:\\data\\DiDiData\\data_csv\\result\\group_pair_result.csv')

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
    od_20_dict = get_20_od_dict()
    # gbrt_pair_result.csv 需要改为 pmp_pair_result.csv
    group_predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# gbrt_pair_result.csv + HA 的結果:
# MSE: 79.3474
# MAE: 3.4326
# MSLE: 0.2978
# r^2 on test data : 0.942694

# pmp_pair_result.csv + HA 的結果:
# MSE: 63.1823
# MAE: 3.3934
# MSLE: 0.3707
# r^2 on test data : 0.957478
