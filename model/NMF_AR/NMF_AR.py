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
    B = nmf.fit_transform(matrix_S)     # B:(58*58行, k列)
    P = nmf.components_                 # P:(k行, 48*24列)
    return B, P


def AR_predict_P(lamb=2):
    pass


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    matrix_S = get_matrix_S()
    matrix_B, matrix_P = matrix_decomposition(matrix_S, k=6)
    AR_predict_P(lamb=2)
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))






