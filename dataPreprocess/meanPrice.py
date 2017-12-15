import os
import pandas as pd
import datetime
from Tools import funcTools as ft

# 某一天 起始地-目的地 对的数量和平均价格, 得到每一天的数据文件

'''
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

dirPath = '/home/zx/data/DiDiData/data_csv/order_data_districtDealed/'
fileList = ft.listdir_nohidden(dirPath)
fileList.sort()
print(fileList)

for i in range(len(fileList)):
    filePath = os.path.join(dirPath, fileList[i])
    fileDate = fileList[i].split('_')[2]
    df_in = pd.read_csv(filePath)

    df_out = df_in.loc[:, ['start_district_id', 'dest_district_id', 'Price']]\
        .sort_values(by=['start_district_id', 'dest_district_id'], axis=0,  ascending=True)

    # df_out.set_index(keys='start_district_id', inplace=True)
    # 根据起始地/目的地id进行分组, 并求其 起始地-目的地 对的数量和平均价格
    size = list(df_out.groupby(['start_district_id', 'dest_district_id']).size())
    df_out = df_out.groupby(['start_district_id', 'dest_district_id']).mean()
    df_out['count'] = size  # 将数量信息写入 df_out 中新的一列


    # 选择某一行或某一方框的值
    # print(type(df_out.loc[1].loc[-1]))
    # print(df_out.loc[1].loc[-1]['Price'])
    # print('df_out.xs((1, -1)):', df_out.xs((1, -1))['count'])
    # print('df_out.xs:', df_out.xs(1, level='start_district_id'))
    # print('df_out.xs:', type(df_out.xs(1, level='start_district_id')))


    index0 = list(df_out.index.levels[0])   # 多重索引中第一索引,len(index0)则为第一索引的长度

    # 选择目的地id 为 -1 的行,能不能以此来删除行?
    # print('df_out.index1:', df_out.xs(-1, level='dest_district_id'))

    # 删除目的地 dest_district_id = -1 的行
    for j in index0:
        curr_index1 = df_out.xs(j, level='start_district_id').index
        for k in curr_index1:
            # print(df_out.xs((j, k)))
            if k == -1:
                df_out.drop((j, k), axis=0, inplace=True)       # 删除的行具有多重索引要加 ()


    destPath = '/home/zx/data/DiDiData/data_csv/Price_mean/price_data_'+fileDate
    df_out.to_csv(destPath)


time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

'''

#--------------------------------------------------------------------------------------------

# 所有数据（不是某一天）起始地-目的地 对的数量和平均价格,混合所有数据得到一个数据文件

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

dirPath = '/home/zx/data/DiDiData/data_csv/order_data_districtDealed/'
fileList = ft.listdir_nohidden(dirPath)
fileList.sort()
print(fileList)

num = 0

for i in range(len(fileList)):
    filePath = os.path.join(dirPath, fileList[i])
    fileDate = fileList[i].split('_')[2]
    df_in = pd.read_csv(filePath)

    df_temp = df_in.loc[:, ['start_district_id', 'dest_district_id', 'Price']]\
        .sort_values(by=['start_district_id', 'dest_district_id'], axis=0,  ascending=True)

    df_temp.set_index(keys=['start_district_id', 'dest_district_id'], inplace=True)

    num = num + len(df_temp)
    print('len: ', len(df_temp))
    print('len: ', num)

    # 合并两个文件的数据
    if i == 0:
        df_out = df_temp
    else:
        df_out = df_out.append(df_temp)

    print(df_out.head(5))

    # 最后一天的数据加入后
    if i == len(fileList)-1:

        df_out.rename(columns={'Price': 'price_mean_total'}, inplace=True)

        # 根据起始地/目的地id进行分组, 并求其 起始地-目的地 对的数量和平均价格
        size = list(df_out.groupby(['start_district_id', 'dest_district_id']).size())
        df_out = df_out.groupby(['start_district_id', 'dest_district_id']).mean()
        df_out['count'] = size  # 将数量信息写入 df_out 中新的一列


        index0 = list(df_out.index.levels[0])   # 多重索引中第一索引,len(index0)则为第一索引的长度

        # 删除目的地 dest_district_id = -1 的行
        for j in index0:
            curr_index1 = df_out.xs(j, level='start_district_id').index
            for k in curr_index1:
                # print(df_out.xs((j, k)))
                if k == -1:
                    df_out.drop((j, k), axis=0, inplace=True)       # 删除的行具有多重索引要加 ()


destPath = '/home/zx/data/DiDiData/data_csv/temp/price_mean_total.csv'
df_out.to_csv(destPath)

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)


