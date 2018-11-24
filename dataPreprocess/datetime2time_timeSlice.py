# 将提取出订单量的csv文件中的TimeBand属性时间片，删除日期
# eg：(2016-02-23 03:00:00, 2016-02-23 03:30:00]  --> (03:00:00, 03:30:00]

import pandas as pd
import Tools.funcTools as ft
import os
from datetime import datetime


'''
# order_count_totalCity（城市总订单量的时间片）文件夹下的csv文件的TimeBand替换
# 3个文件，运行一次替换一个文件

def replaceTimeBand(df):
    for i in range(len(df)):
    # for i in range(5):
        timeBand = df['TimeBand'][i].split(',')
        if len(timeBand[0].split()) == 1:
            start = '(00:00:00'
        else:
            start = '('+timeBand[0].split()[1]
        if len(timeBand[1].split()) == 1:
            end = '24:00:00]'
        else:
            end = timeBand[1].split()[1]
        df['TimeBand'][i] = start+', '+end
        # print(df['TimeBand'][i])
    # print(df['TimeBand'])
    return df

if __name__ == '__main__':
    time_Now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', time_Now)
    filePath = 'E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_10min.csv'
    df = pd.read_csv(filePath)
    print(df.dtypes)
    replaceTimeBand(df)
    df.set_index(keys=['Date', 'TimeBand'], inplace=True)
    destPath = 'E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\totalFlow_10min_removeDate.csv'
    df.to_csv(destPath)
'''


# order_count_15min / 10min / 30min（i -> j 的时间片）文件夹下的csv文件的TimeBand替换
# 3个文件夹，运行一次替换一个文件夹

def readFile(fileList, dirPath):
    for i in range(len(fileList)):
    # for i in range(1):
        filePath = os.path.join(dirPath, fileList[i])
        fileDate = fileList[i].split('_')[2].split('.')[0]  # 2016-02-23

        df = pd.read_csv(filePath)
        replaceTimeBand(df)
        df.set_index(keys=['TimeBand', 'start_district_id', 'dest_district_id'], inplace=True)
        destPath = dirPath+'order_count_removeDate\\'+fileDate + '.csv'
        df.to_csv(destPath)



def replaceTimeBand(df):
    for i in range(len(df)):
    # for i in range(5):
        timeBand = df['TimeBand'][i].split(',')
        if len(timeBand[0].split()) == 1:
            start = '(00:00:00'
        else:
            start = '('+timeBand[0].split()[1]
        if len(timeBand[1].split()) == 1:
            end = '24:00:00]'
        else:
            end = timeBand[1].split()[1]
        df['TimeBand'][i] = start+', '+end
        # print(df['TimeBand'][i])
    # print(df['TimeBand'])
    return df

if __name__ == '__main__':
    time_Now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', time_Now)

    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_count_15min\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    fileList.pop(len(fileList)-1)   # 只取.csv文件，移除dirPath下的order_count_removeDate目录
    readFile(fileList, dirPath)

    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_count_30min\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    fileList.pop(len(fileList)-1)   # 只取.csv文件，移除dirPath下的order_count_removeDate目录
    readFile(fileList, dirPath)

    dirPath = 'E:\\data\\DiDiData\\data_csv\\order_count_60min\\'
    fileList = ft.listdir_nohidden(dirPath)
    fileList.sort()
    fileList.pop(len(fileList)-1)   # 只取.csv文件，移除dirPath下的order_count_removeDate目录
    readFile(fileList, dirPath)

    time_Now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('current time:', time_Now)



