
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

x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 1, 6, 5, 8, 3, 7, 4, 10, 8, 12, 11, 7, 10, 13, 10, 12, 14, 15, 9, 12, 3, 2, 6, 2, 2, 4, 1, 1, 2, 0, 3, 0, 1]
sum = 0
for i in x:
    sum+=i
print(sum)
print(list(map(lambda i:i/sum, x)))
# numerator = -6.77626357803e-21


