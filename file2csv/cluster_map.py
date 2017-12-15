import pandas as pd
import os

dirPath = '/home/zx/data/DiDiData/training_set/cluster_map/'
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
            datalist.append(str)

    columns = ['district_hash', 'district_id']

    print('len(datalist): ', len(datalist))
    df = pd.DataFrame(datalist, columns=columns)
    df.set_index(keys=columns[0], inplace=True)
    df.to_csv('/home/zx/data/DiDiData/data_csv/cluster_map/' + fileName + '.csv')

