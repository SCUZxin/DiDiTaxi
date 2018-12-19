
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
print(x)

x = [[1,2,3],[4,5,6]]
print(2*x)


df_cluster = pd.read_csv('E:\\data\\DiDiData\\data_csv\\cluster_dataset\\time_seg_cluster_result_dict.csv')
for i in range(len(df_cluster)):
    print(df_cluster.loc[i, 'OD_key'], df_cluster.loc[i, 'cluster_num'])

# Pti[ 1 -1] 0.000521376433785
# sum1 : 0.000588755694185
# sum_of_weight_dict[str(t)] : 0.102609125388
# sum2 : 1.12923342145
# sum3 : 5.34979798686e-05

t670_673 = 0.832318124354
x1 = t670_673
t671_673=0.8649580508
x2 = t671_673
t672_673=0.931272267556
x3 = t672_673
print(x1*1*0.1**3+x2*1*0.1**2+x3*1*0.1)

d1 = {'A':123}
print(len(d1))
d1 = {}
print(len(d1))

