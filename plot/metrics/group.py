# 衡量组合预测模型的效果

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# PMWA：     3.2488   7.6516   245   0.3348    0.9638
# PMWA+HA：  3.2696   7.6555   245   0.3512    0.9607

PMWA = [3.2488, 7.6516, 2.45, 3.348, 9.638]
PMWA_HA = [3.2696, 7.6555, 2.45, 3.512, 9.607]

PMWA_real = [3.2488, 7.6516, 245, 0.3348, 0.9638]
PMWA_HA_real = [3.2696, 7.6555, 245, 0.3512, 0.9607]

name_list = ['MAE', 'RMSE', 'MEx$10^{-2}$', 'RMSLEx$10}$', 'R-Squaredx$10$']

plt.figure(figsize=(8, 6), dpi=80)
plt.subplot(1, 1, 1)
x = np.arange(5)
total_width, n = 0.6, 2  # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(left=x + 0.3, height=PMWA, width=width, color='b', label='PMWA')
plt.bar(left=x + width + 0.3, height=PMWA_HA, width=width, color='r', label='PMWA_HA')
for i in range(5):
    if i == 2:
        plt.annotate("%.0f" % PMWA_real[i], xy=(i+0.2, PMWA[i]), xytext=(2, 10), textcoords='offset points')
        plt.annotate("%.0f" % PMWA_HA_real[i], xy=(i + 0.5, PMWA_HA[i]), xytext=(2, 10), textcoords='offset points')
    else:
        plt.annotate("%.4f" % PMWA_real[i], xy=(i, PMWA[i]), xytext=(2, 10), textcoords='offset points')
        plt.annotate("%.4f" % PMWA_HA_real[i], xy=(i + 0.5, PMWA_HA[i]), xytext=(2, 10), textcoords='offset points')

plt.xticks(x + 0.6, name_list, fontsize=12)
plt.xlabel('评价指标')
# plt.ylabel('评价指标值')
plt.legend(loc='best')
plt.show()

