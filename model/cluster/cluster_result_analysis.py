# 对聚类结果进行分析，提取出聚类后每一个簇中包含的OD对
# 画出聚类结果的饼图

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


def analysis_cluster():
    df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result.csv')
    df1 = df_cluster.loc[df_cluster['real_num'] == 1].reset_index(drop=True)
    df2 = df_cluster.loc[df_cluster['real_num'] == 2].reset_index(drop=True)
    df3 = df_cluster.loc[df_cluster['real_num'] == 3].reset_index(drop=True)
    df4 = df_cluster.loc[df_cluster['real_num'] == 4].reset_index(drop=True)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    values_count = df_cluster['real_num'].value_counts()
    print(values_count)
    plot_pie(values_count)
    print('cluster1:')
    for i in range(len(df1)):
        od = str(df1.loc[i, 'start_district_id']) + '-' + str(df1.loc[i, 'dest_district_id'])
        cluster1.append(od)
        print(od, end='，')
    print('\ncluster2:')
    for i in range(len(df2)):
        od = str(df2.loc[i, 'start_district_id']) + '-' + str(df2.loc[i, 'dest_district_id'])
        print(od, end='，')
        cluster2.append(od)
    print('\ncluster3:')
    for i in range(len(df3)):
        od = str(df3.loc[i, 'start_district_id']) + '-' + str(df3.loc[i, 'dest_district_id'])
        print(od, end='，')
        cluster3.append(od)
    print('\ncluster4:')
    for i in range(len(df4)):
        od = str(df4.loc[i, 'start_district_id']) + '-' + str(df4.loc[i, 'dest_district_id'])
        print(od, end='，')
        cluster4.append(od)


# 画出聚类结果的饼图
def plot_pie(values_count):
    print(type(values_count))
    values_count = values_count.sort_index()

    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"), dpi=80)

    data = values_count.values
    categories = ['簇'+str(x) for x in list(values_count.index)]
    # explode = [0, 0, 0.1, 0]
    explode = [0, 0, 0, 0]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),

                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="簇的类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("聚类结果分布")
    plt.show()



if __name__ == '__main__':

    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    analysis_cluster()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 1    113
# 2     88
# 3     88
# 4    255

