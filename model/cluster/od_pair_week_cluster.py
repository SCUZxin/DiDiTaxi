# 先计算每一个OD对在2.23-3.17（除了3.8日）这23天中在工作日和周末的总订单量分别是多少
# 得到 每一个OD对工作日or周末订单量/该OD对所有订单量 的比例向量（2维度，工作日，周末），保存在 od_vector_week_dict.pkl
# 使用比例向量进行聚类


import numpy as np
import pandas as pd

from datetime import datetime
import pickle


# 获取每一个OD对在 (工作日, 周末) 的订单量比例向量
def get_week_vector_of_every_OD():
    try:
        od_vector_week_dict = pickle.load(open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\od_vector_week_dict.pkl', 'rb'))
    except FileNotFoundError:
        df_set = pd.read_csv('E:\\data\\DiDiData\\data_csv\\dataset\\allset.csv')
        df_set = df_set.loc[pd.to_datetime(df_set['date']) != datetime(2016, 3, 8)].reset_index(drop=True)
        df_set['week_day'] = pd.to_datetime(df_set['date']).map(lambda x: x.isoweekday())
        df_set['is_weekend'] = df_set['week_day'].map(lambda x: 1 if x == 6 or x == 7 else 0)

        df_set = df_set.groupby(['start_district_id', 'dest_district_id', 'is_weekend'])['count'].sum()
        df = pd.DataFrame()
        df['start_district_id'] = df_set.index.get_level_values('start_district_id')
        df['dest_district_id'] = df_set.index.get_level_values('dest_district_id')
        df['is_weekend'] = df_set.index.get_level_values('is_weekend')
        df['count'] = list(df_set.values)

        # 得到每一个OD对的向量，保存在 od_vector_day_dict.pkl
        od_vector_week_dict = {}
        for s in range(1, 59):
            for d in range(1, 59):
                od_vector_week_list = [0, 0]
                try:
                    is_weekend = list(df_set.xs((s, d)).index.values)
                    count_vector = list(df_set.xs((s, d)).values)
                    for i in range(len(is_weekend)):
                        od_vector_week_list[is_weekend[i]] = count_vector[i]
                    sum1 = np.array(od_vector_week_list).sum()
                    prop_vector = list(map(lambda x: x / sum1, od_vector_week_list))
                    od_vector_week_dict[str(s)+'-'+str(d)] = prop_vector
                except KeyError:
                    continue
        pickle.dump(od_vector_week_dict, open('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\od_vector_week_dict.pkl', 'wb'))
    return od_vector_week_dict


# 使用knn对是否是周末进行聚类
def is_weekend_cluster_kmeans():
    # 看看dict的values中有没有重复的？有重复的，只是值重复，地址不重复
    global od_vector_week_dict
    keys = od_vector_week_dict.keys()
    values = list(od_vector_week_dict.values())     # 随机的顺序

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabaz_score
    import matplotlib.pyplot as plt

    kmeans_model = KMeans(n_clusters = 10, random_state = 1,init='k-means++')
    pred = kmeans_model.fit_predict(values)
    print(pred)
    print(kmeans_model.cluster_centers_) #聚类的中心
    print(kmeans_model.labels_)#每个样本所属的簇
    print(kmeans_model.inertia_)

    i = []
    y_silhouette_score = []
    inertia_score = []
    calinskiharabaz_score = []
    for k in range(2, 20):
        print(k)
        kmeans_model = KMeans(n_clusters=k, random_state=1, init='k-means++')
        pred = kmeans_model.fit_predict(values)
        silhouettescore = silhouette_score(values, pred)
        # print("silhouette_score for cluster '{}'".format(k))
        # print(silhouettescore)
        calinskiharabazscore = calinski_harabaz_score(values, pred)
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




if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    od_vector_week_dict = get_week_vector_of_every_OD()
    is_weekend_cluster_kmeans()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

