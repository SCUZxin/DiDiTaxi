# 按照时间顺序产生起止对的比例矩阵，将所有矩阵按照时间顺序传入list中，写入 proportion.pkl 文件
# prop_list:0-815, 1151

import pandas as pd
import numpy as np

from datetime import datetime
import pickle


# 获取订单量比例矩阵的 list 集合，并 return
def get_proportion():
    try:
        prop_list = pickle.load(open('E:\\data\\DiDiData\\data_csv\\dataset\\proportion_list5858.pkl', 'rb'))
        # np_data = prop_list[51]
        # np.savetxt("E:\\data\\DiDiData\\data_csv\\dataset\\result.txt", np_data)
        return prop_list
    except FileNotFoundError:
        df_pair_data = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
        df_total_data = pd.read_csv('E:\\data\\DiDiData\\data_csv\\order_count_totalCity\\'
                                    'totalFlow_30min_replaceTimeBand.csv')
        prop_list = []
        for i in range(len(df_total_data)):
        # for i in range(1):
        #     i = 51
            if i % 100 == 0:
                print('iterator  ', i)
            prop_array = np.zeros((58, 58))     # 方便以区域id作为下标index
            date = df_total_data.loc[i, 'date']
            time = df_total_data.loc[i, 'half_hour']
            total_count = df_total_data.loc[i, 'count']
            df_time_slice = df_pair_data.loc[(df_pair_data['date'] == date) & (df_pair_data['time'] == time)]\
                .reset_index(drop=True)
            for j in range(len(df_time_slice)):
            # for j in range(10):
                start_id = df_time_slice.loc[j, 'start_district_id']
                dest_id = df_time_slice.loc[j, 'dest_district_id']
                count = df_time_slice.loc[j, 'count']
                prop_array[start_id-1][dest_id-1] = count/total_count
            prop_list.append(prop_array)
        pickle.dump(prop_list, open('E:\\data\\DiDiData\\data_csv\\dataset\\proportion_list5858.pkl', 'wb'))
        return prop_list


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    proportion_list = get_proportion()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


