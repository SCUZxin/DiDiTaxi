# 统计一级POI类别一共有多少个,有的是 1#10:83   24:581， 没有#号的表示二级类别为其它
# 结论：区域一级类别为：1—25,
# level=9的POI所有区域都没有，保存到文件poi_1st_level_count_dict.pkl中，
# key是区域id（1-58）的str， value是level：1-25的POI数量的list(index：0-24)


import os
import pandas as pd
import numpy as np
import datetime
from Tools import funcTools as ft
import pickle


def poi_count():
    file_path = 'E:\\data\\DiDiData\\data_csv\\poi_data_districtDealed\\poi_data.csv'
    df = pd.read_csv(file_path)

    # 20# 7:332-20#5:249-20#4:83-20#2:664-20#1:913-16#12:913-16#10:166-16#11:332-20#8:830-
    # 23#3:83-23#5:83-5#3:83-22#5:249-22#4:498-14#6:332-19#3:1079-25:83-19#1:913-25#9:249-
    # 20:332-19#4:166-23:166-25#7:249-1:83-4:166-7:1079-6:415-24#3:83-6#4:166-24#1:249-
    # 17#5:83-15#6:83-4#17:83-15#3:332-15#2:415-5#1:415-8#4:332-4#7:249-4#2:664-4#1:83-
    # 8#2:166-4#11:83-4#12:332-16#3:249-16#4:498-4#16:83-1#10:249-4#18:166-17#2:166-3#1:2324-
    # 1#6:249-11:83-1#2:83-15:415-14:249-1#5:166-16:498-19:6640-2#5:83-1#9:249-1#8:166-
    # 14#10:83-2#2:83-11#8:1162-21#2:581-2#10:83-21#1:83-13#4:166-19#2:83-11#5:83-11#4:249

    poi_dict = {}
    try:
        poi_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\features\\poi_1st_level_count_dict.pkl', 'rb'))
        # print(poi_dict['1'])
    except FileNotFoundError:
        for i in range(len(df)):
            poi_np = np.zeros(25)
            # for i in range(1):
            poi_data = df.loc[i, 'poi_class']
            poi_list = poi_data.split('-')
            for j in range(len(poi_list)):
                if '#' in poi_list[j]:
                    first_level = poi_list[j].split('#')[0]
                else:
                    first_level = poi_list[j].split(':')[0]
                first_level_count = int(poi_list[j].split(':')[1])
                poi_np[int(first_level)-1] += first_level_count
                key = str(i+1)
                poi_dict[key] = poi_np

        pickle.dump(poi_dict, open('E:\\data\\DiDiData\\data_csv\\features\\poi_1st_level_count_dict.pkl','wb'))
    return poi_dict


if __name__ == '__main__':
    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    poi_count_dict = {}
    poi_count_dict = poi_count()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



