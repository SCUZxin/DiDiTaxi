# 提取交通的特征相似度：
# 1. 得到每个区域的 tj3 + tj4 之和，对每个区域，找到所有时间片中的 max 和 min
#    保存到 traffic_max_min_dict.pkl, key是区域 id （1-58），value: [max, min]
# 2. 计算每个区域目标时间片（counter=673-1152）与历史（closeness(pre_len=1) 和 period（pre_len=1））的相似度
#    保存到 traffic_similarity_dict.pkl  key：区域id，两个时间片的counter（eg：58:96-1152）value: 相似度

import pandas as pd
from datetime import datetime
import pickle


# 计算每个区域目标时间片（counter=673-1152）与历史（closeness(pre_len=1) 和 period（pre_len=1））的相似度
# 保存到 traffic_similarity_dict.pkl  key：区域id，两个时间片的counter（eg：58:96-1152）value: 相似度
def get_traffic_similarity_dict():
    try:
        # traffic_max_min_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\dataset\\traffic_max_min_dict.pkl', 'rb'))
        traffic_similarity_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\dataset\\traffic_similarity_dict.pkl', 'rb'))
    except FileNotFoundError:
        df_traffic = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\traffic_feature.csv')
        df_traffic['tc_count'] = df_traffic['tj_level_3'] + df_traffic['tj_level_4']
        # print(len(df_traffic))  # 66815，应该是66816啊,2.29, time=9, id=54
        # print(df_traffic['district_id'].value_counts())   # 1-58区域都是：1152
        for i in range(1, 5):
            del df_traffic['tj_level_'+str(i)]
        df_max_min = df_traffic.groupby(['district_id'])
        df_traffic_max = list(df_max_min['tc_count'].max().values)
        df_traffic_min = list(df_max_min['tc_count'].min().values)
        traffic_max_min_dict = {}
        for i in range(len(df_traffic_max)):
            max_min_list = []
            max_min_list.append(df_traffic_max[i])
            max_min_list.append(df_traffic_min[i])
            traffic_max_min_dict[str(i+1)] = max_min_list
        # print(traffic_max_min_dict)

        df_similarity = df_traffic.groupby(['district_id', 'date', 'time'])['tc_count'].sum()
        df = pd.DataFrame()
        df['district_id'] = df_similarity.index.get_level_values('district_id')
        df['date'] = df_similarity.index.get_level_values('date')
        df['time'] = df_similarity.index.get_level_values('time')
        df['count'] = list(df_similarity.values)
        # df = df.loc[0:383].append(df.loc[768:1151]).append(df.loc[384:767]).reset_index(drop=True)
        df = df.sort_values(by=['district_id', 'date', 'time'], axis=0, ascending=True)
        print(df.loc[380:385])

        values_list = list(df['count'].values)

        tc_all_region_list = []      # 保存1-58区域的每个 counter 的 tc_list
        for i in range(58):
            tc_list = []
            for j in range(1152):
                tc_list.append(values_list[i*1152+j])
            tc_all_region_list.append(tc_list)

        print(tc_all_region_list[0][380:390])

        traffic_similarity_dict = {}
        # 计算可能的相似度并保存
        for id in range(1,59):
            max = traffic_max_min_dict[str(id)][0]
            min = traffic_max_min_dict[str(id)][1]
            for counter in range(673,1153):
                # 上几周该时间片 period
                period_time_list = []
                ind = counter
                while ind > 7 * 48:
                    ind = ind - 7 * 48
                    period_time_list.append(ind)
                his_time_list = period_time_list+[counter-3, counter-2, counter-1]    # 要用来和 counter 进行比较相似度的历史时间片
                for t in his_time_list:
                    tc_counter = tc_all_region_list[id-1][counter-1]
                    tc_his_counter = tc_all_region_list[id-1][t-1]
                    print(id)
                    print(t)
                    print(tc_counter)
                    print(tc_his_counter)
                    similary = 1 - (abs(tc_counter-tc_his_counter)/(max-min))
                    traffic_similarity_dict[str(id)+':'+str(t)+'-'+str(counter)] = similary

        # pickle.dump(traffic_max_min_dict, open('E:\\data\\DiDiData\\data_csv\\dataset\\traffic_max_min_dict.pkl', 'wb'))
        # pickle.dump(traffic_similarity_dict, open('E:\\data\\DiDiData\\data_csv\\dataset\\traffic_similarity_dict.pkl', 'wb'))
    print(traffic_max_min_dict['1'])
    print(traffic_similarity_dict['1:670-673'])
    return traffic_similarity_dict

if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    get_traffic_similarity_dict()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))






