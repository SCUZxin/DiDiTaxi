'''
以区域 id 来替换文件中的区域 hash 值
'''

import pandas as pd
import os

districtPath = '/home/zx/data/DiDiData/data_csv/cluster_map/cluster_map.csv'
df_district = pd.read_csv(districtPath)

# 对区域定义表的区域 hash 与 id 保存为 dict
id_dict = dict()
for i in range(len(df_district)):
    id_dict[df_district.district_hash[i]] = df_district.district_id[i]

'''

# ----------------------------------------------------------------------------
# 以区域id替换文件夹 poi_data 中文件的区域 hash 值

poiPath = '/home/zx/data/DiDiData/data_csv/poi_data/poi_data.csv'
df_poi = pd.read_csv(poiPath)
df_poi['district_id'] = None
for i in range(len(df_poi)):
    district_id = id_dict.get(df_poi.district_hash[i], -1)  # 根据 hash 得到对应的 id
    df_poi.at[i, 'district_id'] = district_id   # df_poi.district_id[i] 也可

df_out = df_poi.iloc[:, 1:].sort_values(by='district_id', axis=0, ascending=True)     # 只取后两列并排序
# print(df_out.columns)
df_out.set_index(keys='district_id', inplace=True)  # 以区域 id 为索引

print(df_out.head())
desPath = '/home/zx/data/DiDiData/data_csv/poi_data_districtDealed/poi_data.csv'
df_out.to_csv(desPath)
# -------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# 以区域id替换文件夹 traffic_data 中文件的区域 hash 值

traffic_dirPath = '/home/zx/data/DiDiData/data_csv/traffic_data/'
traffic_fileList = os.listdir(traffic_dirPath)
for i in range(len(traffic_fileList)):
    filePath = os.path.join(traffic_dirPath, traffic_fileList[i])
    df_traffic = pd.read_csv(filePath)
    df_traffic['district_id'] = None
    for j in range(len(df_traffic)):
        district_id = id_dict.get(df_traffic.district_hash[j], -1)  # 根据 hash 得到对应的 id
        df_traffic.at[j, 'district_id'] = district_id  # df_traffic.district_id[i] 也可

    df_out = df_traffic.iloc[:, 1:].sort_values(by=['district_id', 'tj_time'], axis=0, ascending=True)  # 只取后3列并排序
    # print(df_out.columns)
    df_out.set_index(keys='district_id', inplace=True)  # 以区域 id 为索引

    print(df_out.head())
    desPath = '/home/zx/data/DiDiData/data_csv/traffic_data_districtDealed/'+traffic_fileList[i]
    df_out.to_csv(desPath)
# ----------------------------------------------------------------------------------------
'''


# ----------------------------------------------------------------------------
# 以区域id替换文件夹 order_data 中文件的区域 hash 值

order_dirPath = '/home/zx/data/DiDiData/data_csv/order_data/'
order_fileList = os.listdir(order_dirPath)
order_fileList.sort()

for i in range(len(order_fileList)):
    filePath = os.path.join(order_dirPath, order_fileList[i])
    print(order_fileList[i])
    df_order = pd.read_csv(filePath)
    df_order.rename(columns={'start_district_hash': 'start_district_id', 'dest_district_hash': 'dest_district_id'}, inplace=True)

    for j in range(len(df_order)):
        # 重命名区域hash 的列名
        df_order.at[j, 'start_district_id'] = id_dict.get(df_order.start_district_id[j], -1)  # df_order.start_district_id[i] 也可
        df_order.at[j, 'dest_district_id'] = id_dict.get(df_order.dest_district_id[j], -1)  # df_order.dest_district_id[i] 也可


    df_order.set_index(keys='order_id', inplace=True)
    print(df_order.head())
    desPath = '/home/zx/data/DiDiData/data_csv/order_data_districtDealed/'+order_fileList[i]
    df_order.to_csv(desPath)
# ----------------------------------------------------------------------------------------


