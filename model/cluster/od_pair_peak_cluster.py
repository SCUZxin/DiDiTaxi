# 只看工作日的流量
# 只划分流量多的20%的 OD 对，使用4段时间的比例向量（v1, v2, v3, v4）, 然后 K-means 聚类 ？？？
# 分割时间片为 t1, t2, t3, t4, 得到早高峰时段，早~晚时段，晚高峰时段，其它时段
# 聚类结果保存到 time_seg_cluster_result_dict ， key: OD 对（'5-3'）,value：聚类所属的簇


# 根据   晚后早前、早高峰(7:30-10:00, [15,19])、早后晚前、晚高峰(17:00-19:30, [34, 38])
# 找到一天中流量最大的两个时间片，看是否处于(7:30-9:30, [15,18])和(17:00-19:00, [34, 37])
# 且值大于（11:00-15:00 [22, 29]的平均值的 2 倍）
# 四段的流量进行分类，分为：
# 有早有晚、有早无晚、无早有晚、无早无晚

import numpy as np
import pandas as pd

from datetime import datetime
import pickle


def load_data(t1, t2, t3, t4):
    try:
        time_seg_vector_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_vector_dict.pkl', 'rb'))
    except FileNotFoundError:
        df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
        df_set = df_set.loc[pd.to_datetime(df_set['date']) != datetime(2016, 3, 8)].reset_index(drop=True)
        df_set['week_day'] = pd.to_datetime(df_set['date']).map(lambda x: x.isoweekday())
        df_set['is_weekend'] = df_set['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)
        df_set = df_set.loc[df_set['is_weekend'] == 0].reset_index(drop=True)
        early_peak = [x for x in range(t1, t2)]
        between_ealy_late = [x for x in range(t2, t3)]
        late_peak = [x for x in range(t3, t4)]
        other = [x for x in range(48) if x not in early_peak+between_ealy_late+late_peak]

        df_set['time_seg'] = df_set['time'].map(lambda x: 1 if x in early_peak
        else(2 if x in between_ealy_late else(3 if x in late_peak else(4 if x in other else -1))))

        df_set = df_set.groupby(['start_district_id', 'dest_district_id', 'time_seg'])['count'].sum()
        df = pd.DataFrame()
        df['start_district_id'] = df_set.index.get_level_values('start_district_id')
        df['dest_district_id'] = df_set.index.get_level_values('dest_district_id')
        df['time_seg'] = df_set.index.get_level_values('time_seg')
        df['count'] = list(df_set.values)

        # return
        od_below_80perc_list = pickle.load(open('E:\\data\\DiDiData\\data_csv\\plot\\od_below_80perc_list.pkl', 'rb'))

        # 得到每一个OD对的向量，保存在 od_vector_hour_dict.pkl
        time_seg_vector_dict = {}
        for s in range(1, 59):
            for d in range(1, 59):
                key = str(s)+'-'+str(d)
                if key not in od_below_80perc_list:
                    time_vector_seg_list = [0] * 4
                    try:
                        time_seg_list = list(df_set.xs((s, d)).index.values)
                        # print(time_seg_list)
                        count_vector = list(df_set.xs((s, d)).values)
                        for i in range(len(time_seg_list)):
                            time_vector_seg_list[time_seg_list[i]-1] = count_vector[i]
                        sum1 = np.array(time_vector_seg_list).sum()
                        prop_vector = list(map(lambda x: x / sum1, time_vector_seg_list))
                        time_seg_vector_dict[key] = prop_vector
                    except KeyError:
                        continue
        pickle.dump(time_seg_vector_dict,
                    open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_vector_dict.pkl', 'wb'))
    return time_seg_vector_dict


# 使用k-Means对 time_seg_vector_dict 进行聚类
def time_seg_cluster_kmeans():
    try:
        df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result_dict.csv')
    except FileNotFoundError:
        # 看看dict的values中有没有重复的？有重复的，只是值重复，地址不重复
        global time_seg_vector_dict
        key_list = []
        value_list = []
        for key in time_seg_vector_dict:
            key_list.append(key)
            value_list.append(time_seg_vector_dict[key])

        print(key_list)
        print(value_list)

        # values = list(time_seg_vector_dict.values())  # 随机的顺序

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.metrics import calinski_harabaz_score
        import matplotlib.pyplot as plt

        kmeans_model = KMeans(n_clusters=4, random_state=1, init='k-means++')
        pred = kmeans_model.fit_predict(value_list)

        time_seg_cluster_result_list = kmeans_model.labels_
        df_cluster = pd.DataFrame({'OD_key': key_list,
                                   'cluster_num': time_seg_cluster_result_list})
        df_cluster.to_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result_dict.csv')

        print(pred)
        print(kmeans_model.cluster_centers_)  # 聚类的中心
        print(kmeans_model.labels_)  # 每个样本所属的簇
        print(kmeans_model.inertia_)

    return df_cluster

'''
        i = []
        y_silhouette_score = []
        inertia_score = []
        calinskiharabaz_score = []
        for k in range(2, 20):
            print(k)
            kmeans_model = KMeans(n_clusters=k, random_state=1, init='k-means++')
            pred = kmeans_model.fit_predict(value_list)
            silhouettescore = silhouette_score(value_list, pred)
            # print("silhouette_score for cluster '{}'".format(k))
            # print(silhouettescore)
            calinskiharabazscore = calinski_harabaz_score(value_list, pred)
            # print("calinski_harabaz_score '{}'".format(k))
            # print(calinskiharabazscore)
            i.append(k)
            y_silhouette_score.append(silhouettescore)
            inertia_score.append(kmeans_model.inertia_)
            calinskiharabaz_score.append(calinskiharabazscore)
            # print("kmeans_model.inertia_score for cluster '{}'".format(k))
            # print(kmeans_model.inertia_)

        # 转成字典方便查找
        # dict_silhouette = dict(zip( i,y_silhouette_score))
        # dict_inertia_score = dict(zip( i,inertia_score))
        # dict_calinskiharabaz_score = dict(zip( i, calinskiharabaz_score))

        plt.figure()
        plt.plot(i, y_silhouette_score)
        plt.xlabel("kmeans-k")
        plt.ylabel("silhouette_score")
        plt.title("matrix")

        plt.figure()
        plt.plot(i, inertia_score)
        plt.xlabel("kmeans-k")
        plt.ylabel("inertia_score(sum of squared)")
        plt.title("matrix")

        plt.figure()
        plt.plot(i, calinskiharabaz_score)
        plt.xlabel("kmeans-k")
        plt.ylabel("calinski_harabaz_score")
        plt.title("matrix")
        plt.show()
'''

if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    time_seg_vector_dict = load_data(15, 19, 34, 38)
    time_seg_cluster_kmeans()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


