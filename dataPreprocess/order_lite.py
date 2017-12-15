# 将区域id替换过的订单数据进行处理,只保留'order_id', 'start_district_id',
#  'dest_district_id', 'Price','Time'五项,并按照起始地区域id 排序
# 目的：查看price, Time 更直观方便

from Tools import funcTools as ft
import os
import pandas as pd
import datetime

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

dirPath = '/home/zx/data/DiDiData/data_csv/order_data_districtDealed/'
fileList = ft.listdir_nohidden(dirPath)
fileList.sort()
print(fileList)

for i in range(len(fileList)):
# for i in range(1):
    filePath = os.path.join(dirPath, fileList[i])
    fileDate = fileList[i].split('_')[2]
    df_in = pd.read_csv(filePath)

    df_out = df_in.loc[:, ['order_id', 'start_district_id', 'dest_district_id', 'Price', 'Time']]\
        .sort_values(by=['start_district_id', 'dest_district_id'], axis=0,  ascending=True)

    df_out.set_index(keys='order_id', inplace=True)

    destPath = '/home/zx/data/DiDiData/data_csv/order_lite/order_'+fileDate
    df_out.to_csv(destPath)


time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)