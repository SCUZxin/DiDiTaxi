# dest_district_id 不存在(dest_district_id=-1)的订单量
'''
import pandas as pd
filePath = '/home/zx/data/DiDiData/data_csv/order_data_districtDealed/order_data_2016-02-23.csv'
df = pd.read_csv(filePath)
count = 0
for i in range(len(df)):
    if df.at[i, 'dest_district_id'] == -1:
        count += 1
print('count:', count)  # 4602  # 41342
'''

# 74c1c25f4b283fa74a5514307b0d0278
# c4bfad9b76cae464126ab1030ab63a62
# dd8d3b9665536d6e05b29c2648c0e69a
# 364bf755f9b270f0f9141d1a61de43ee
# c4ec24e0a58ebedaa1661e5c09e47bb5
# 8786db08e9a725579c52428a00eeb64a
# 08232402614a9b48895cc3d0aeb0e9f2
# d1ab2cc538d518758a1a82b1787592d4





# import pandas as pd
# Path = '/home/zx/data/DiDiData/data_csv/temp/price_mean_total.csv'
# df = pd.read_csv(Path)
#
# count = 0
# for i in range(len(df)):
#     count += df.at[i, 'count']
#
# print(count)



# import h5py as h5
#
# f = h5.File('/home/zx/Python/PycharmProjects/DeepST-master/data/TaxiBJ/BJ_Meteorology.h5')
# print(pd.read_hdf(filePath, 'df'))


import pandas as pd
file = '/home/zx/Python/PycharmProjects/DeepST-master/data/TaxiBJ/BJ_Meteorology.h5'
# print(pd.read_hdf(file, 'df'))

