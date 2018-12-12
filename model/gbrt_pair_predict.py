# 使用 GBRT 模型预测多目标起止对的的订单量, 使用 GradientBoostingRegression

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


def save_model(filename, m):
    pickle.dump({'model': m}, open(filename+'.pkl', 'wb'))


def save_result(filename, yp):
    global x_test
    result = pd.DataFrame()
    # result['test_id'] = range(len(yp))
    result['date'] = date
    result['time'] = x_test['time']
    result['start_district_id'] = x_test['start_district_id']
    result['dest_district_id'] = x_test['dest_district_id']
    result['count'] = yp
    result.to_csv(filename+'.csv', index=False)


# 评价指标为MAE(mean absolute error), 即预测值与真实值的绝对误差，并取平均数
def mae_average(y_pred, y_name):
    # assert len(y_predict) == len(y_name)
    mae_sum = 0.0
    for j in range(len(y_pred)):
        mae_sum += abs(y_pred[j] - y_name[j])
    return mae_sum * 1.0 / len(y_pred)


def gen_data_set():
    df_train = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Train.csv')
    df_test = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\Test.csv')
    global date
    date = df_test['date']
    del df_train['date']
    del df_test['date']
    return df_train, df_test


def gen_model():
    # Fit regression model
    n_estimators = [i for i in range(100, 1000, 100)]
    max_depth = [i for i in range(3, 8, 1)]
    min_samples_split = [i for i in range(2, 10, 1)]
    learning_rate = [i for i in np.arange(0.25, 0.28, 0.01)]
    fit_model()
    # for i in range(len(n_estimators)):
    #     for j in range(len(learning_rate)):
    #         mse = fit_model(n_estimators[i], learning_rate[j])
    #         print(n_estimators[i], '   %.2f'%(learning_rate[j]), '  mse:', mse)
    # print(mse.values)


def fit_model(n_estimators=500, learning_rate=0.26):
    params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 200,
              'learning_rate': 0.1, 'verbose': 2, 'loss': 'ls', 'random_state': 0}
    # params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'verbose': 1, 'loss': 'ls', 'random_state': 0}
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'loss': 'ls'}
    # n_estimators: 500, 'max_depth': 5, 'min_samples_split': 6,
    # learning_rate: 0.26,
    # gbrt = ensemble.GradientBoostingRegressor(**params)
    gbrt = ensemble.GradientBoostingRegressor(verbose=2)
    # gbrt = XGBRegressor(**params)

    # gbrt = ensemble.GradientBoostingRegressor(loss='ls',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2, min_samples_leaf = 256, min_samples_split = 256)
    fileName = 'E:\\data\\DiDiData\\data_csv\\result\\gbrt_pair_result'

    global x_train, y_train

    # 进行归一化之后

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    x_train_minmax = x_scaler.fit_transform(x_train)
    x_test_minmax = x_scaler.transform(x_test)
    y_train_minmax = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    gbrt.fit(x_train_minmax, y_train_minmax)
    y_predict_minmax = gbrt.predict(x_test_minmax)
    y_predict = y_scaler.inverse_transform(y_predict_minmax.reshape(-1, 1))

    # save_result(fileName, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    msle = mean_squared_log_error(y_test, y_predict)
    print("MSLE: %.4f" % msle)  # 输出 mean_squared_log_error
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


    # 归一化之前

    # # gbrt.fit(x, y)
    # # gbrt.fit(x_test, y_test)
    # gbrt.fit(x_train, y_train)
    # save_model(fileName, gbrt)
    # model_result = gbrt.predict(x_test)
    # save_result(fileName, list(model_result))
    # mse = mean_squared_error(y_test, gbrt.predict(x_test))
    # print("MSE: %.4f" % mse)  # 输出均方误差
    # mae = mean_absolute_error(y_test, model_result)
    # print("MAE: %.4f" % mae)  # 输出平均绝对误差
    # r2 = r2_score(y_test, model_result)
    # print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好
    # # return mse

    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(gbrt.staged_predict(x_test)):
        test_score[i] = gbrt.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, gbrt.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # Plot feature importance
    feature_importance = gbrt.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    columns = x_train.columns
    plt.yticks(pos, columns[sorted_idx])
    # plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    date = pd.DataFrame()
    train, test = gen_data_set()
    x_train, y_train = train.iloc[:, 0:-1], train.loc[:, 'count']
    x_test, y_test = test.iloc[:, 0:-1], test.loc[:, 'count']

    for i in range(1, 26):
        start_name = 'start_poi_'+str(i)+'_count'
        dest_name = 'dest_poi_'+str(i)+'_count'
        del x_train[start_name]
        del x_train[dest_name]
        del x_test[start_name]
        del x_test[dest_name]

    # del x_train['start_district_id']
    # del x_train['dest_district_id']
    # del x_test['start_district_id']
    # del x_test['dest_district_id']

    gen_model()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 归一化之前
# default para
# MSE: 79.3915
# r^2 on test data : 0.950966

# 归一化之后 default para
# MSE: 78.1567
# MAE: 3.2887
# r^2 on test data : 0.951729

# 归一化之后，删除50个 POI 特征 default para
# MSE: 77.1199
# MAE: 3.2752
# r^2 on test data : 0.952369

# 归一化之后，删除50个 POI 特征，删除起止地 id default para
# MSE: 76.9486
# MAE: 3.2752
# r^2 on test data : 0.952475

# 归一化之后，删除50个 POI 特征
# params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 200,
#           'learning_rate': 0.1, 'verbose': 2, 'loss': 'ls', 'random_state': 0}
# MSE: 82.4942
# MAE: 3.3465
# r^2 on test data : 0.949050

