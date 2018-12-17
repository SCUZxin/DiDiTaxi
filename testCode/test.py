
import numpy as np
import pandas as pd
import pickle

x1 = np.zeros((3, 3))
x2 = np.array([[11,12,13],[14,15,16],[17,18,19]])
y = [x1, x2]


# with open('E:\\data\\DiDiData\\data_csv\\result\\pmp_pair_para_result.txt', 'a') as f:
#     f.write('MSE    MAE     MLSE    r^2\n')
print('xxxxxxxxxxx')
from sklearn.metrics import mean_squared_error

x = [1,2,3]
y = [4,5,6]
print(x+y)
print(list(map(lambda i:i/2, x)))

df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result_dict.csv')
for i in range(len(df_cluster)):
    print(df_cluster.loc[i, 'OD_key'], df_cluster.loc[i, 'cluster_num'])



