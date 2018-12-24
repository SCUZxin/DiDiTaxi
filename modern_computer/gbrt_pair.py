# 使用 GBRT   来预测 OD 对的流量 80%HA， 20% GBRT（一个80%图）
# 对比：HA、卡尔曼滤波、SVR、ARIMA
# 只拿一个 OD 对（要求：每个时刻都有流量）来训练 ARIMA、卡尔曼滤波、SVR
# 图：某一天的流量变化图，是不是更接近，累积误差是不是最小（暂时不用）
# 柱状图：几个评价指标选一个或者几个最好的


from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# from pyramid.arima import auto_arima

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from datetime import datetime


# 获取训练集和测试集的数据
def get_all_data(start=9, dest=9, day=datetime(2016, 3, 11)):
    df_test = pd.read_csv('E:\\data\\DiDiData\\data_csv\\result\\gbrt_pair_result.csv')
    df_test = df_test.loc[(df_test['start_district_id'] == start) & (df_test['dest_district_id'] == dest)]
    # 分成训练集和测试集
    # df_train = df_allset.loc[pd.to_datetime(df_allset['date']) <= datetime(2016, 3, 10)].reset_index(drop=True)
    # df_test = df_allset.loc[pd.to_datetime(df_allset['date']) > datetime(2016, 3, 10)].reset_index(drop=True)

    # 先构造训练集，如果某时间片为0，则 count 置为0
    # df_train['date'] = pd.to_datetime(df_train['date'])
    # df_test['date'] = pd.to_datetime(df_test['date'])
    # df_train = df_train.loc[(df_train['start_district_id']==start) & (df_train['dest_district_id']==dest)]
    # df_test = df_test.loc[(df_test['start_district_id']==start) & (df_test['dest_district_id']==dest)]
    # date_list = pd.date_range('2016/2/23', '2016/3/10')
    # df_train_final = pd.DataFrame()
    # for date in date_list:
    #     for i in range(48):
    #         df_temp = df_train.loc[(df_train['date']==date) & (df_train['time']==i)]
    #         if len(df_temp)==0:
    #             df_temp = pd.DataFrame(columns=['start_district_id', 'dest_district_id','date','time','count'])
    #             df_temp.loc[0] = [start, dest, date, i, 0]
    #         df_train_final = df_train_final.append(df_temp).reset_index(drop=True)
    # print(len(df_train_final))
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test = df_test.loc[(df_test['date'] == day)]
    df_test_final = pd.DataFrame()
    for i in range(48):
        df_temp = df_test.loc[df_test['time'] == i]
        if len(df_temp) == 0:
            df_temp = pd.DataFrame(columns=['date', 'time', 'start_district_id', 'dest_district_id',
                                            'predict_count', 'real_count'])
            df_temp.loc[0] = [day, i, start, dest, 0, 0]
        df_test_final = df_test_final.append(df_temp).reset_index(drop=True)
    print(len(df_test_final))
    # return df_train_final, df_test_final
    return df_test_final


def svr_predict():
    from sklearn import svm
    global df_train, df_test
    train = df_train['count']
    del df_train['date']
    del df_train['count']
    test = df_test['count']
    del df_test['date']
    del df_test['count']
    rbf = svm.SVR(kernel='rbf')
    rbf.fit(df_train, train)
    y_predict = rbf.predict(df_test)
    # print(y_predict)

    rmse = sqrt(mean_squared_error(y_predict, test))
    print("RMSE: %.4f" % rmse)  # 输出均方误差
    mae = mean_absolute_error(y_predict, test)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(y_predict, test)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_predict, test)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


def plot_flow_change():
    global df_test
    print(df_test)
    x = range(1, 49)
    y_predict = list(df_test['predict_count'].values)
    y_real = list(df_test['real_count'].values)
    # plt.xlabel('订单量', fontproperties=font)
    # plt.ylabel('起止对流量比例', fontproperties=font)
    plt.title('date: 2016-3-11    OD-pair: 9-9')
    plt.xlabel('时间片')
    plt.ylabel('流量值')
    plt.plot(x, y_predict, color='red', marker='^', linestyle='--', linewidth=2, label='GBRT')
    plt.plot(x, y_real, color='black', marker='o',  ms=10, linestyle='-', linewidth=2, label='真实值')
    plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.show()


def plot_metrics():
    # 顺序分别是ha，svr, arima， gbrt
    MSE = [60.9486, 82.4942, 78.1567, 111.8363]
    MAE = [2.5407, 3.3465, 3.2887, 3.7116]
    RMSE = list(map(lambda x: sqrt(x), MSE))
    print(RMSE)
    name_list = ['GBRT', 'SVR', 'ARIMA', 'HA']


    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)
    x = np.arange(4)
    total_width, n = 0.6, 2  # 有多少个类型，只需更改n即可
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(left=x+0.3, height=RMSE, width=width,  color='b', label='RMSE')
    plt.bar(left=x+width+0.3, height=MAE, width=width, color='r', label='MAE')
    plt.xticks(x+0.6, name_list, fontsize=12)
    plt.xlabel('预测算法')
    # plt.ylabel('评价指标值')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start, dest = 9, 9
    predict_day = datetime(2016, 3, 11)
    df_test = get_all_data(start, dest, predict_day)
    plot_flow_change()
    plot_metrics()
    # arima_predict()
    # svr_predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))




