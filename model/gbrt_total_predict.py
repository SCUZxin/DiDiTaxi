# 使用 GBRT 模型预测城市总的订单量, 使用 GradientBoostingRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle


def save_model(filename, m):
    pickle.dump({'model': m}, open(filename+'.pkl', 'wb'))


def save_result(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(len(yp))
    result['count'] = yp
    result.to_csv(filename+'.csv', index=False)


# 评价指标为MAE(mean absolute error), 即预测值与真实值的绝对误差，并取平均数
def mae_average(y_pred, y_name):
    # assert len(y_predict) == len(y_name)
    mae_sum = 0.0
    for j in range(len(y_pred)):
        mae_sum += abs(y_pred[j] - y_name[j])
    return mae_sum * 1.0 / len(y_pred)


def gen_total_set():
    df_weather = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\weather_feature.csv')
    df_time = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\time_feature.csv')
    # df_time = df_time[['week_day', 'day', 'is_weekend', 'is_vocation', 'lagging_3', 'lagging_2', 'lagging_1', 'count']]
    del df_time['date']
    del df_time['time']
    df = pd.concat([df_weather, df_time], axis=1)
    x, y = df.iloc[:, 1:-1], df.loc[:, 'count']
    return x, y


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
    # MSE: 652009.6635                  default: MSE: 639654.8721
    # r ^ 2 on test data: 0.969268      default: r^2 on test data : 0.969850
    params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 7,
              'learning_rate': 0.26, 'verbose': 0, 'loss': 'ls', 'random_state': 0}
    # params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'verbose': 1, 'loss': 'ls', 'random_state': 0}
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'loss': 'ls'}
    # n_estimators: 500, 'max_depth': 5, 'min_samples_split': 6,
    # learning_rate: 0.26,
    gbrt = ensemble.GradientBoostingRegressor(**params)
    # gbrt = ensemble.GradientBoostingRegressor(verbose=0)
    # gbrt = XGBRegressor(**params)

    # gbrt = ensemble.GradientBoostingRegressor(loss='ls',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2, min_samples_leaf = 256, min_samples_split = 256)
    fileName = 'E:\\data\\DiDiData\\data_csv\\result\\gbrt_toal_result'

    global x_train, y_train

    # 进行归一化之后的预测

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    x_train_minmax = x_scaler.fit_transform(x_train)
    x_test_minmax = x_scaler.transform(x_test)
    y_train_minmax = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    gbrt.fit(x_train_minmax, y_train_minmax)
    y_predict_minmax = gbrt.predict(x_test_minmax)
    y_predict = y_scaler.inverse_transform(y_predict_minmax.reshape(-1, 1))

    save_result(fileName, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)  # 输出均方误差
    mae = mean_absolute_error(y_test, y_predict)
    print("MAE: %.4f" % mae)  # 输出平均绝对误差
    r2 = r2_score(y_test, y_predict)
    print("r^2 on test data : %f" % r2)  # R^2 拟合优度=(预测值-均值)^2之和/(真实值-均值)^2之和,越接近1越好


    # 没有进行归一化之前的代码，也没有对weather, week_day进行one-hot

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
    x, y = gen_total_set()
    x_train, y_train = x.iloc[0:int(len(x)/24*17), :], y[0:int(len(y)/24*17)]
    x_test, y_test = x.iloc[int(len(x)/24*17):len(x), :], y[int(len(y)/24*17):len(y)]

    # del x_train['pm2.5_change']
    # del x_test['pm2.5_change']
    # del x_train['pm2.5_mean_day']
    # del x_test['pm2.5_mean_day']
    gen_model()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



# 归一化前
#     params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 7,
#               'learning_rate': 0.26, 'verbose': 0, 'loss': 'ls', 'random_state': 0}
# MSE: 652009.6635                  default: MSE: 639654.8721
# r ^ 2 on test data: 0.969268      default: r^2 on test data : 0.969850

# [0, 1] 归一化后,默认参数
# MSE: 1006796.3809
# MAE: 674.5548
# r^2 on test data : 0.952545

