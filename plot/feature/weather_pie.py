# 统计天气情况的分布，用饼状图

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import numpy as np
import pandas as pd
from datetime import datetime


# 画出聚类结果的饼图
def plot_weather_pie():
    df = pd.read_csv('E:\\data\\DiDiData\\data_csv\\features\\before\\weather_feature.csv')
    print(df.head())
    values_count = df['weather'].value_counts()

    print(type(values_count))
    print(values_count)     # 2,1,3,4
    weather_map = {'1': '晴', '2': '多云', '3': '阴', '4': '雨'}
    values_count = values_count.sort_index()

    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"), dpi=80)

    data = values_count.values
    categories = [weather_map[str(x)] for x in list(values_count.index)]
    # explode = [0, 0, 0.1, 0]
    explode = [0, 0, 0, 0.1]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="天气", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("天气情况分布")
    plt.show()


if __name__ == '__main__':
    print('start time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    plot_weather_pie()
    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

