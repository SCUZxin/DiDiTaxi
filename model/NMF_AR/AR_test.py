
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# from pyramid.arima import auto_arima
from statsmodels.tsa.ar_model import AR

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
    df_allset = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    df_allset['date'] = pd.to_datetime(df_allset['date'])
    # 分成训练集和测试集
    df_train = df_allset.loc[pd.to_datetime(df_allset['date']) <= datetime(2016, 3, 10)].reset_index(drop=True)
    df_test = df_allset.loc[pd.to_datetime(df_allset['date']) > datetime(2016, 3, 10)].reset_index(drop=True)

    # 先构造训练集，如果某时间片为0，则 count 置为0
    df_train = df_train.loc[(df_train['start_district_id']==start) & (df_train['dest_district_id']==dest)]
    df_test = df_test.loc[(df_test['start_district_id']==start) & (df_test['dest_district_id']==dest)]

    date_list = pd.date_range('2016/2/23', '2016/3/10')
    df_train_final = pd.DataFrame()
    for date in date_list:
        for i in range(48):
            df_temp = df_train.loc[(df_train['date']==date) & (df_train['time']==i)]
            if len(df_temp)==0:
                df_temp = pd.DataFrame(columns=['start_district_id', 'dest_district_id','date','time','count'])
                df_temp.loc[0] = [start, dest, date, i, 0]
            df_train_final = df_train_final.append(df_temp).reset_index(drop=True)
    # print(df_train_final.tail())
    # print(len(df_train_final))

    df_test = df_test.loc[(df_test['date'] == day)]
    df_test_final = pd.DataFrame()
    for i in range(48):
        df_temp = df_test.loc[df_test['time'] == i]
        if len(df_temp) == 0:
            df_temp = pd.DataFrame(columns=['start_district_id', 'dest_district_id','date','time','count'])
            df_temp.loc[0] = [start, dest, day, i, 0]
        df_test_final = df_test_final.append(df_temp).reset_index(drop=True)
    # print(df_test_final)
    # print(len(df_test_final))

    return df_train_final, df_test_final



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


def ar_mode():
    global df_train, df_test
    X = list(df_train['count'].values)
    train, test = X, list(df_test['count'].values)
    # train autoregression
    model = AR(train)
    # model_fit = model.fit(maxlag=1)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # print('len(coef)', len(coef))
    # print('coef', coef)
    # print('window', window)
    # walk forward over time steps in test
    history = train[len(train) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    print('test', test)
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]
        # yhat = 0
        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    mse = mean_squared_error(test, predictions)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(test, predictions)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(test, predictions)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(test, predictions)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()




if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start, dest = 9, 9
    predict_day = datetime(2016, 3, 11)
    df_train, df_test = get_all_data(start, dest, predict_day)
    ar_mode()
    # plot_flow_change()
    # plot_metrics()
    # arima_predict()
    # svr_predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))






