import pandas as pd
import datetime
import os

# 当前时间
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

# 根据目录名得到该目录下文件的列属性,相当于switch 功能
def columnsDecision(args):
    switcher = {
        'cluster_map': ['district_hash', 'district_id'],
        'order_data': ['order_id', 'driver_id', 'passenger_id', 'start_district_hash',
                       'dest_district_hash', 'Price', 'Time'],
        'poi_data': ['district_hash', 'poi_class'],
        'traffic_data': ['district_hash', 'tj_level', 'tj_time'],
        'weather_data': ['Time', 'Weather', 'temperature', 'PM2.5']
    }
    return switcher.get(args)


dirPath = '/home/zx/data/DiDiData/training_set/traffic_data/'
fileList = os.listdir(dirPath)  # 得到目录下的文件名列表
fileList.sort()     # 得到的文件列表按文件名排序

# 对目录下所有的文件进行操作（遍历以及写成csv）
for i in range(len(fileList)):
    fileName = fileList[i]
    filePath = os.path.join("%s%s" % (dirPath, fileList[i]))  # 某一文件的绝对路径
    datalist = []
    # 读文件
    with open(filePath, 'r') as f:
        for line in f.readlines():
            str = line.strip().split('\t')
            str_tj_level = '-'.join(str[1:5])
            strtemp = [str[0], str_tj_level, str[5]]
            datalist.append(strtemp)

    # print(datalist[0])
    columns = ['district_hash', 'tj_level', 'tj_time']

    print('len(datalist): ', len(datalist))
    df = pd.DataFrame(datalist, columns=columns)
    df.set_index(keys=columns[0], inplace=True)
    df.to_csv('/home/zx/data/DiDiData/data_csv/traffic_data/' + fileName + '.csv')

