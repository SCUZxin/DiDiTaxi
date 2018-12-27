
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import math
import pickle


# with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
#     f.write('MSE    MAE     MLSE    r^2\n')
print('xxxxxxxxxxx')
from sklearn.metrics import mean_squared_error

x = [[1,2,3,5],[1,5,9],[1,1,1],[2,6]]
y = [[4,5,6,9],[2,1,5],[2,2,2],[1,9]]
print(x+y)
# print(list(map(lambda i:i/2, x)))
error_m_list = list(map(lambda x: list(map(lambda x1: x1[0]+x1[1], zip(x[0], x[1]))), zip(x, y)))
print(error_m_list)

df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result.csv')

print(df_cluster['cluster_num'].value_counts())
print(df_cluster['real_num'].value_counts())

x = [np.array(1.1), np.array(3.6)]
print(x)
print(x[0])
x = list(map(lambda i: round(float(i)), x))
print(x)


from numpy.linalg import *

x = np.asarray([3,5])
y = np.asarray([6,1])
dist = norm(x - y)
print(dist)
x = [14,9,6,10,15]
print(x[-1])
y = sorted(x)
for i in x:
    print(y.index(i))

