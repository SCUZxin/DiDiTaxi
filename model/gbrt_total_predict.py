# 使用 GBRT 模型预测城市总的订单量, 使用 GradientBoostingRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
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
    df_time = df_time[['week_day', 'day', 'is_weekend', 'is_vocation', 'lagging_3', 'lagging_2', 'lagging_1', 'count']]
    df = pd.concat([df_weather, df_time], axis=1)
    x, y = df.iloc[:, 2:-1], df.loc[:, 'count']
    return x, y


if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    X, Y = gen_total_set()
    x_train, y_train = X.iloc[0:int(len(X)/24*17), :], Y[0:int(len(Y)/24*17)]
    x_test, y_test = X.iloc[int(len(X)/24*17):len(X), :], Y[int(len(Y)/24*17):len(Y)]

    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
              'learning_rate': 0.01, 'verbose': 1, 'loss': 'ls', 'random_state': 0}
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'loss': 'ls'}
    gbrt = ensemble.GradientBoostingRegressor(**params)
    # gbrt = XGBRegressor(**params)

    # gbrt = ensemble.GradientBoostingRegressor(loss='ls',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2, min_samples_leaf = 256, min_samples_split = 256)
    fileName = 'E:\\data\\DiDiData\\data_csv\\result\\gbrt_result'

    # gbrt.fit(X, Y)
    gbrt.fit(x_train, y_train)
    save_model(fileName, gbrt)
    model_result = gbrt.predict(x_test)
    save_result(fileName, list(model_result))
    mse = mean_squared_error(y_test, gbrt.predict(x_test))
    print("MSE: %.4f" % mse)

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

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


