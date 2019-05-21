# 代码运行时间对比

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

time_list = [14, 33, 35, 43, 108, 153]
name_list = ['HA', 'PMWA-HA', 'GBRT', 'NMF-AR', 'PMWA', 'STP-KNN']

plt.figure(figsize=(8, 6), dpi=80)
plt.subplot(1, 1, 1)
x = np.arange(6)
x = np.array([0.2, 0.7, 1.2, 1.7, 2.2, 2.7])
total_width, n = 0.2, 1  # 有多少个类型，只需更改n即可
width = total_width / n
# x = x - (total_width - width) / 2
x = x - 0.1
plt.bar(left=x, height=time_list, width=width, color='b')
# plt.bar(left=x + width + 0.3, height=PMWA_HA, width=width, color='r', label='PMWA_HA')
for i in range(6):
        plt.annotate("%d" % time_list[i], xy=(x[i]+0.02, time_list[i]), xytext=(2, 10), textcoords='offset points')

plt.xticks(x+0.1, name_list, fontsize=12)
plt.xlabel('模型')
plt.ylim(0, 200)
plt.ylabel('运行时间(s)')
plt.legend(loc='best')
plt.show()





