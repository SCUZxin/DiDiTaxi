# 统计一共有多少个区域的hash Id
# 结论：start_district_hash 所有数据都是只有58个
#       dest_district_hash 所有数据都不止58个，一般超过300个
# 因此，需要删除 dest_district_hash 不在 cluster_map 中那58个的订单

import os
import pandas as pd
import datetime
from Tools import funcTools as ft


def read_files_keep_hash(file_list):
    for i in range(len(file_list)):
    # for i in range(2):
        file_path = os.path.join(dirPath, file_list[i])
        file_date = file_list[i].split('_')[2]
        df = pd.read_csv(file_path)

        df = df.loc[:, ['start_district_hash', 'dest_district_hash', 'Price', 'Time']] \
            .sort_values(by=['start_district_hash', 'dest_district_hash'], axis=0,  ascending=True)

        # df.set_index(keys=['start_district_hash', 'dest_district_hash', 'Time'], inplace=True)

        start_count = df['start_district_hash'].nunique()
        dest_count = df['dest_district_hash'].nunique()
        if start_count != 58:
            print('start_count:', file_date, start_count)
        if dest_count != 58:
            print('dest_count:', file_date, dest_count)

        if i == len(file_list)-1:
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('current time:', end_time)


if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', start_time)

    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_data\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    read_files_keep_hash(fileList)


