# 区域 A、B之间的时滞影响，从A到B的order_count和从B到A的 order_count之间的影响
# 思路：X1=25,X2=20时，df_Ax1_Bx2只有46行，根据 str(X_time[i]) 获取df_Ax1_Bx2 的对应行，不能获取，
# 则置对应行的count为0，然后根据A->B 和 B->A 的order_count 画图，比较折线图

from datetime import datetime, date, timedelta
import pandas as pd

time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('current time:', time)

today = date(2016, 3, 6)
tomorrow = today + timedelta(days=1)
str_today = today.isoformat()
str_tomorrow = tomorrow.isoformat()

filePath = 'E:\\data\\DiDiData\\data_csv\\order_count\\order_count_'+str_today+'.csv'
df = pd.read_csv(filePath)

# A、B分别是x1=25和x2=20
x1 = 16
x2 = 44
df_Ax1_Bx2 = df[(df['start_district_id'] == x1) & (df['dest_district_id'] == x2)]
df_Ax2_Bx1 = df[(df['start_district_id'] == x2) & (df['dest_district_id'] == x1)]
# df_Ax1_Bx2 = df.loc[(df['start_district_id'] == 25) & (df['dest_district_id'] == 20)]
# df_Ax1_Bx2.sort_values(by='TimeBand', axis=0,  ascending=True)
df_Ax1_Bx2.set_index(keys='TimeBand', inplace=True)
df_Ax2_Bx1.set_index(keys='TimeBand', inplace=True)
print(df_Ax1_Bx2.columns)   # Index(['start_district_id', 'dest_district_id', 'count'], dtype='object')


X_time = pd.interval_range(start=pd.Timestamp(str_today+' 00:00:00'),
                           end=pd.Timestamp(str_tomorrow+' 00:00:00'), freq='0.5H')

for i in range(len(X_time)):
    time_slice = str(X_time[i]) # eg: (2016-02-23 05:00:00, 2016-02-23 05:30:00]
    try:
        lineX1_X2 = df_Ax1_Bx2.loc[time_slice]  # 根据时间片来选择对应行，目的用于查看该时间片是否有订单
    except KeyError as e:   # 出异常，即该行不存在，将其添加到行中，count置为0
        print("keyError! can't get the line of corresponding time slice!")
        # pd.to_datetime(time_slice)
        df_Ax1_Bx2.loc[time_slice] = [x1, x2, 0]
    finally:
        pass

    try:
        lineX2_X1 = df_Ax2_Bx1.loc[time_slice]  # 根据时间片来选择对应行，目的用于查看该时间片是否有订单
    except KeyError as e:   # 出异常，即该行不存在，将其添加到行中，count置为0
        print("keyError! can't get the line of corresponding time slice!")
        # pd.to_datetime(time_slice)
        df_Ax2_Bx1.loc[time_slice] = [x2, x1, 0]
    finally:
        pass

# 排序后 (2016-02-23, 2016-02-23 00:30:00] 到第一行了，切片，将最后一行移至第一行
df_Ax1_Bx2 = df_Ax1_Bx2.sort_index()
length = len(df_Ax1_Bx2)
df_Ax1_Bx2 = df_Ax1_Bx2.iloc[length-1:, :].append(df_Ax1_Bx2.iloc[0:length-1])

# 排序后 (2016-02-23, 2016-02-23 00:30:00] 到第一行了，切片，将最后一行移至第一行
df_Ax2_Bx1 = df_Ax2_Bx1.sort_index()
length = len(df_Ax2_Bx1)
df_Ax2_Bx1 = df_Ax2_Bx1.iloc[length-1:, :].append(df_Ax2_Bx1.iloc[0:length-1])

# dirPath = 'E:\\data\\DiDiData\\data_csv\\temp\\tempX1_X2.csv'
# df_Ax1_Bx2.to_csv(dirPath)
# dirPath = 'E:\\data\\DiDiData\\data_csv\\temp\\tempX2_X1.csv'
# df_Ax2_Bx1.to_csv(dirPath)
# print(type(df_Ax1_Bx2.loc[:, ['TimeBand']]))


# 根据A->B 和 B->A 的order_count 画图，比较折线图

import matplotlib.pyplot as plt
import numpy as np
X = np.array(df_Ax1_Bx2.index)
Y_Ax1_Bx2 = np.array(df_Ax1_Bx2['count'])
Y_Ax2_Bx1 = np.array(df_Ax2_Bx1['count'])
print(X)
print(type(X))
print(Y_Ax1_Bx2)
print(type(Y_Ax1_Bx2))
plt.plot(np.arange(48), Y_Ax1_Bx2, color='red', linestyle='--', linewidth=2.0, label='x1=%d->x2=%d' % (x1, x2))
plt.plot(np.arange(48), Y_Ax2_Bx1, color='blue', linestyle='--', linewidth=2.0, label='x2=%d->x2=%d' % (x2, x1))

# plt.plot(X, np.arange(48), color='red')
plt.legend(loc="upper right")
plt.show()




