#test task: 对数据导入、特征选择、特征展示、聚类分析及展示进行测试
from pyEnergy.cluster import kmeans
from pyEnergy.fool import initialize_with_feature_selector, initialize
from pyEnergy.drawer import draw_signal, draw_corr, draw_parallel_coordinates
import matplotlib.pyplot as plt

fool = initialize("data/ChangErZhai-40-139079-values 20180101-20181031.csv")
fool.feature_selection(method="pca", selection_params={"n_components": 3})
# draw_corr(feature=fool.feature)
y_pred = kmeans(*fool.features(), plot=True)
draw_parallel_coordinates(fool.feature, y_pred)
fool.box()
# fig, ax = plt.subplots(2, 2)
# ax0 = ax[0][0]
# ax1 = ax[1][0]
# ax2 = ax[0][1]
# ax3 = ax[1][1]
draw_signal(fool.other_event[0])
# draw_signal(fool.other_event[1], ax=ax1)
# draw_signal(fool.other_event[-2], ax=ax2)
# draw_signal(fool.other_event[-1], ax=ax3)
# plt.show()


