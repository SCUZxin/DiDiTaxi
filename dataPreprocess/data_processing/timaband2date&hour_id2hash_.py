# 1.将data_csv\order_count_30min\order_count_removeDate中csv文件的id根据cluster_map.csv 转为 hash值
# id --> 这部分可以暂时不做，就先以id进行训练
# 2.将TimeBand 列转为date 和 half_hour(30min为时间片的话：0-47), 包括order_count_30min\order_count_removeDate
#   文件夹下的和 order_count_totalCity 文件夹下的
# 3.在 data_csv\order_count_30min\order_count_1 目录生成一个总的csv文件， order_count_totalCity下生成totalCity的

import os
import pandas as pd
import datetime
from Tools import funcTools as ft


# 将时间片与dict类型的half_hour进行匹配
def match_half_hour(time_band):
    if time_band not in half_hour:
        half_hour[time_band] = len(half_hour)
    return half_hour[time_band]


# 将所有csv文件的时间片转为half_hour, 同时加上date 列
def add_info_pair(file, time_interval_para='30', is_match=False):
    file_path = os.path.join(dirPath, file)
    file_date = file.split('.')[0]
    df = pd.read_csv(file_path)

    # 添加 date 和 half_hour 字段，删除 TimeBand 字段
    df_drop_dupli = df.drop_duplicates(['TimeBand'])
    time_bands = df_drop_dupli['TimeBand'].values

    if is_match:  # 只需要第一次进行dict匹配就行
        # 匹配 half_hour = {}
        for j in range(len(time_bands)):
            match_half_hour(time_bands[j])
    df['date'] = file_date
    df['half_hour'] = df['TimeBand'].apply(lambda x: half_hour[x])
    del df['TimeBand']

    '''
    # 将 start_district_id 改为 start_district_hash， dest_district_id 改为 dest_district_hash，改字段名和值
    file_path = 'E:\\data\\DiDiData\\data_csv\\cluster_map\\cluster_map.csv'
    df_cluster_map = pd.read_csv(file_path, dtype={'district_id': str})
    df[['start_district_id', 'dest_district_id']] = df[['start_district_id','dest_district_id']].astype(str)
    print(type(df_cluster_map.loc[0, 'district_id']))
    print(type(df.loc[0, 'start_district_id']))
    print(type(df.loc[0, 'dest_district_id']))
    df['start_district_id'].replace(df_cluster_map['district_id'].values,
                                    df_cluster_map['district_hash'].values, inplace=True)
    df['dest_district_id'].replace(df_cluster_map['district_id'].values,
                                   df_cluster_map['district_hash'].values, inplace=True)
    df.rename(columns={'start_district_id': 'start_district_hash', 'dest_district_hash': 'dest_district_hash'})
    '''

    df = df[['start_district_id', 'dest_district_id', 'date', 'half_hour', 'count']]

    df.set_index(keys=['start_district_id', 'dest_district_id'], inplace=True)
    dest_path = 'E:\\data\\DiDiData\\data_csv\\order_count_'+time_interval_para+'min\\order_count_replaceTimeBand\\'+file_date+'.csv'
    df.to_csv(dest_path)


def add_info_total(file_path, time_interval_para='30'):
    df = pd.read_csv(file_path)
    # 添加 date 和 half_hour 字段，删除 TimeBand 字段
    df_drop_dupli = df.drop_duplicates(['TimeBand'])
    time_bands = df_drop_dupli['TimeBand'].values

    # # 匹配 half_hour = {}
    # for j in range(len(time_bands)):
    #     match_half_hour(time_bands[j])
    df['half_hour'] = df['TimeBand'].apply(lambda x: half_hour[x])
    del df['TimeBand']

    df = df.rename(columns={'Date': 'date'})
    df = df[['date', 'half_hour', 'count']]

    df.set_index(keys=['date', 'half_hour'], inplace=True)
    dest_path = 'E:\data\DiDiData\data_csv\order_count_totalCity\\totalFlow_'+str(time_interval)+'min_replaceTimeBand.csv'
    df.to_csv(dest_path)


if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start time:', start_time)

    time_interval = '30'

    # 起止对的订单量文件信息更改 timeband -> date、half_hour
    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_count_'+str(time_interval)+'min\\order_count_removeDate'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    half_hour = {}
    match = False
    for i in range(len(fileList)):
        # for i in range(1):
        if i == 0:
            match = True
        add_info_pair(fileList[i], time_interval, match)

    # 城市总的订单量文件信息更改 timeband -> date、half_hour
    filePath = 'E:\data\DiDiData\data_csv\order_count_totalCity\\totalFlow_'+str(time_interval)+'min_removeDate.csv'
    add_info_total(filePath, time_interval)

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end time:', end_time)



