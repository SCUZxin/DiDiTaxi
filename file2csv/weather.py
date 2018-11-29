import pandas as pd
import os

dirPath = 'E:\\data\\DiDiData\\training_set\\weather_data\\'
fileList = os.listdir(dirPath)  # 得到目录下的文件名列表
fileList.sort()     # 得到的文件列表按文件名排序

# 对目录下所有的文件进行操作（遍历以及写成csv）
for i in range(len(fileList)):
    fileName = fileList[i]
    filePath = os.path.join("%s%s" % (dirPath, fileList[i]))  # 某一文件的绝对路径
    date = []
    time = []
    weather = []
    temperature = []
    pm25 = []
    # 读文件
    with open(filePath, 'r') as f:
        for line in f:
            contents = line.split()
            if len(contents) == 5:
                date.append(contents[0])
                time.append(contents[1])
                weather.append(contents[2])
                temperature.append(contents[3])
                pm25.append(contents[4])

    weather_dict = {'date': date,
                    'time': time,
                    'weather': weather,
                    'temperature': temperature,
                    'pm2.5': pm25}

    df = pd.DataFrame(weather_dict)
    df = df[['date', 'time', 'weather', 'temperature', 'pm2.5']]

    df.set_index(keys=['date', 'time'], inplace=True)
    df.to_csv('E:\\data\\DiDiData\\data_csv\\weather_data\\' + fileName + '.csv')

