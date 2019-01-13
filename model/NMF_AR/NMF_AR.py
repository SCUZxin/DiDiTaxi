# 非负矩阵分解 + 自回归


from math import sqrt
import numpy as np
from numpy.linalg import *
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# from pyramid.arima import auto_arima
from statsmodels.tsa.ar_model import AR
from sklearn.decomposition import NMF

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle


def mean_absolute_perc_error(y_predict, y_real):
    sum1 = 0
    for i in range(len(y_predict)):
        sum1 += abs(y_real[i]-y_predict[i]) / y_real[i]
    return sum1/len(y_predict)


# 获取数据并生成矩阵array S（58*58=3364行（0-57是起始地id是1的，依次类推），48*24=1152列）
def get_matrix_S():
    try:
        matrix_S = pickle.load(open('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\matrix_S.pkl', 'rb'))
    except FileNotFoundError:
        df_allset = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
        df_allset['day'] = pd.to_datetime(df_allset['date']).map(lambda x: (x - datetime(2016, 2, 22)).days)
        len_df_allset = len(df_allset)
        # print(df_allset.head())
        matrix_S = np.zeros((58*58, 48*24))
        for i in range(len_df_allset):
            if i % 10000 == 0:
                print('iterator: ', i)
            # 获取这一行的所有值 start_district_id, dest_district_id, date, time, count, day
            value = list(df_allset.loc[i].values)
            matrix_S[(value[0]-1)*58+(value[1]-1)][48*(value[5]-1)+value[3]] = value[4]
        pickle.dump(matrix_S, open('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\matrix_S.pkl', 'wb'))
    return matrix_S


def matrix_decomposition(S, k=6):
    nmf = NMF(n_components=k, random_state=0)
    B = nmf.fit_transform(matrix_S)     # B:(58*58行, k列), (3364, 6)
    P = nmf.components_                 # P:(k行, 48*24列), (6, 1152)
    return B, P


def AR_predict_P_all(lamb=2):
    global matrix_B, matrix_P
    predictions = []
    for i in range(matrix_P.shape[0]):
    # for i in range(1):
        matrix_Pi = list(matrix_P[i])   # 矩阵 P 的每一行，均有1152个元素
        predict_Pi = AR_predict_P_each(matrix_Pi, lamb=lamb)
        predictions.append(predict_Pi)
    predictions = np.asarray(predictions)   # (6, 336) 的数组

    # 订单量的计算
    matrix_result_list = []
    for i in range(predictions.shape[1]):
        P_t = predictions[:, i]
        matrix_result_list.append(list(matrix_B.dot(P_t)))
    return matrix_result_list


def AR_predict_P_each(matrix_Pi, lamb=2):
    train, test = matrix_Pi[0:816], matrix_Pi[816:1152]

    # train autoregression
    model = AR(train)
    # model_fit = model.fit(maxlag=2)
    model_fit = model.fit(maxlag=lamb)
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train) - window:]
    history = [history[i] for i in range(len(history))]
    prediction = list()
    # print('test', test)
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]
        # yhat = 0
        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]
        obs = test[t]
        prediction.append(yhat)
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
    return prediction


def predict():
    global result_list
    df_allset = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
    df_test = df_allset.loc[pd.to_datetime(df_allset['date']) > datetime(2016, 3, 10)].reset_index(drop=True)
    df_test['day'] = pd.to_datetime(df_test['date']).map(lambda x: (x - datetime(2016, 3, 11)).days)
    y_test = list(df_test['count'].values)
    y_predict = []
    for i in range(len(df_test)):
        if i % 10000 == 0:
            print('iterator: ', i)
        # 获取这一行的所有值 start_district_id, dest_district_id, date, time, count, day
        value = list(df_test.loc[i].values)
        matrix_St = result_list[value[5]*48+value[3]]
        predict_value = matrix_St[(value[0] - 1) * 58 + (value[1] - 1)]
        y_predict.append(predict_value)

    y_predict = list(map(lambda x: round(x), y_predict))
    # 如果遇到负数转为正数，免得计算MSLE出错
    y_predict = list(map(lambda x: -x if x < 0 else x, y_predict))
    df_test['predict_count'] = y_predict
    df_test.to_csv('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_result.csv')

    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    mape = mean_absolute_perc_error(y_predict, y_test)
    print("MAPE: %.4f" % mape)  # 输出平均百分比绝对误差
    me = max(list(map(lambda x1,x2:abs(x1-x2), y_predict, y_test)))
    print("ME: %.4f" % me)  # 输出最大误差
    msle = mean_squared_log_error(y_test, y_predict)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好

    with open('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_para.txt', 'a') as f:
        f.write('%.4f' % mse+'      %.4f' % mae+'       %.4f' % msle+'      %f' % r2+'\n')

    #
    # # plot
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    matrix_S = get_matrix_S()
    # matrix_B, matrix_P = matrix_decomposition(matrix_S, k=6)
    # result_list = AR_predict_P_all(lamb=2)
    # predict()
    for k in range(2, 21):
        for lamb in range(1, 21):
            with open('E:\\data\\DiDiData\\data_csv\\result\\NMF-AR\\NMF_AR_para.txt', 'a') as f:
                f.write('k='+ str(k) + '  lambda=' + str(lamb) + ' :\n')
            matrix_B, matrix_P = matrix_decomposition(matrix_S, k=k)
            result_list = AR_predict_P_all(lamb=lamb)
            predict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 直接用分解后的矩阵B（不是预测的）得到的结果，NMF 分解有误差，可能还不小，原因未知
# MSE: 43.5306
# MAE: 3.0109
# MSLE: 0.2907
# r^2 on test data : 0.973115


# k=6, lambda=2
# MSE: 98.5348
# MAE: 3.8616
# MSLE: 0.3295
# r^2 on test data : 0.939143

# k=5  lambda=20 :
# 89.9530      3.7145       0.3299      0.944443

# k=6  lambda=20 :
# 87.6683      3.6781       0.3242      0.945854

# k=8  lambda=20 :
# 86.2063      3.6510       0.3229      0.946757

# k=9  lambda=20 :
# 84.9684      3.6368       0.3201      0.947522

# k=10  lambda=20 :
# 83.4495      3.6166       0.3192      0.948460

# k=20  lambda=20 :
# 86.8412      3.6099       0.3129      0.946365


