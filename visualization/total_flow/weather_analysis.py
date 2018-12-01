# 分析天气数据的分布，各个天气下订单量的多少等?取影响较大的区域，而不是整体流量？ 这点还没做

import pandas as pd
import numpy as np
import Tools.funcTools as ft
import os
from datetime import datetime


# 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
def weather_analysis(file_list, time_interval_para=30):
    pass


if __name__ == '__main__':

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start time:', start_time)

    # dirPath = '/home/zx/data/DiDiData/data_csv/order_lite/'
    dirPath = 'E:\\data\\DiDiData\\data_csv\\weather_data\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()

    # 以30min为时间间隔，将30min内的天气 weather, temperature, pm2.5 提取总结出来，并写入 weather.csv文件
    weather_analysis(fileList, 30)

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end time:', end_time)

