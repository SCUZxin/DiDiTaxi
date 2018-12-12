# 1. 计算出工作日周末的平均比例矩阵avg_prop_array，得到48*2个矩阵
# 2. 使用工作日和周末的总流量的平均值来乘比例矩阵，得到每个的流量
# 3. 看一下误差是多少

import extract_feature.gen_proportion as gen_proportion
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from datetime import datetime
import pickle


def get_avg_prop_list():
    proportion_list = gen_proportion.get_proportion()
    # 0-815 是前17天的
    # 周末   ([5, 6, 12, 13] - 1)*48+(0……47)
    # 工作日 ([1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17] - 1)*48+(0……47)
    weekend_prop_list = []
    weekend = [4, 5, 11, 12]
    for i in range(48):
        tmp = np.zeros((59, 59))
        for j in weekend:
            pos = j*48+i
            tmp += proportion_list[pos]
        tmp = tmp / 4
        weekend_prop_list.append(tmp)

    weekday_prop_list = []
    weekday = np.array([0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16])
    for i in range(48):
        tmp = np.zeros((59, 59))
        for j in weekday:
            pos = j*48+i
            tmp += proportion_list[pos]
        tmp = tmp / 13
        weekday_prop_list.append(tmp)

    return weekday_prop_list, weekend_prop_list


def cal_pair():
    try:
        count_list = pickle.load(open('E:\\data\\DiDiData\\data_csv\\result\\gbrt_total_ha_prop.pkl', 'rb'))
    except FileNotFoundError:
        weekday_prop_list, weekend_prop_list = get_avg_prop_list()
        df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\gbrt_toal_result.csv')
        df_ha_pair = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\ha_pair.csv')
        df_ha_pair['week_day'] = pd.to_datetime(df_ha_pair['date']).map(lambda x: x.isoweekday())
        df_ha_pair['is_weekend'] = df_ha_pair['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)
        y_test = df_ha_pair['count'].values
        count_list = []
        for i in range(len(df_ha_pair)):
        # for i in range(10):
            if i % 10000 == 0:
                print('iterator ', i)
            s = df_ha_pair.loc[i, 'start_district_id']
            d = df_ha_pair.loc[i, 'dest_district_id']
            date = df_ha_pair.loc[i, 'date']
            time = df_ha_pair.loc[i, 'time']
            is_weekend = df_ha_pair.loc[i, 'is_weekend']
            # total = df.loc[(pd.to_datetime(df['date']) == date) & (df['time'] == time)]['mean_his_day'].values
            total = df.loc[(pd.to_datetime(df['date']) == date) & (df['time'] == time)]['count'].values

            if is_weekend:
                count_cal = total * weekend_prop_list[time][s][d]
            else:
                count_cal = total * weekday_prop_list[time][s][d]
            count_list.append(count_cal)

        pickle.dump(count_list, open('E:\\data\\DiDiData\\data_csv\\result\\gbrt_total_ha_prop.pkl', 'wb'))

    mse = mean_squared_error(y_test, count_list)
    print("MSE: %.4f" % mse)  # 输出均方误差
    MAE = mean_absolute_error(y_test, count_list)
    print("MAE: %.4f" % MAE)  # 输出平均绝对误差
    r2 = r2_score(y_test, count_list)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    cal_pair()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# ha_total * proportion平均
# MSE: 113.0120
# MAE: 3.7150
# r^2 on test data : 0.930201
# 相比 ha_pair , MSE 偏大，r^2 更接近 1

# gbrt_total * proportion平均
# MSE: 84.9302
# MAE: 3.3659
# r^2 on test data : 0.947545
# 相比 gbrt_pair， MSE，MAE，r^2 都表现更差，原因：误差累积（total一个误差，proportion一个误差）？？？

