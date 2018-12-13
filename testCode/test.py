
import numpy as np
import pandas as pd
import pickle

x1 = np.zeros((3, 3))
x2 = np.array([[11,12,13],[14,15,16],[17,18,19]])
y = [x1, x2]

df = pd.DataFrame()
df1 = pd.DataFrame({'A':[1,2,3,4,5],
                    'B':['a','b','c','d','e']})
df2 = pd.DataFrame({'A':[6,7,8,9,10],
                    'B':['a','b','c','d','e']})

print(df1)
df.append(df1)
df = pd.concat([df, df1])
df = pd.concat([df, df2]).reset_index(drop=True)
# df.append(df2)
print(df)





