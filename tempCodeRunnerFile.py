from pyEnergy.cluster import kmeans, kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.fool import initialize_with_feature_selector
from pyEnergy.drawer import draw_corr

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(path, method="pca", selection_params={"n_components": 6, "threshold": 0.5})
fool.box()
#? fool.feature_info = [fool.feature.columns[1], fool.feature.columns[9]]
#? fool.feature = fool.feature[[fool.feature.columns[1], fool.feature.columns[9]]] # 使用1和9（对应matlab 2,10）
#? draw_corr(feature=fool.feature) # 绘制相关图 发现相关系数高达0.98
###* ----------feature selection end--------------------
###* ----------model selection start--------------------
# y_pred = kmeans_elbow1(*fool.features(), plot=True, repeat=50, penal=0)
# fool.feature_info = [fool.feature.columns[0],fool.feature.columns[1],fool.feature.columns[2], fool.feature.columns[3]]
# fool.feature = fool.feature[fool.feature_info]
# draw_corr(feature=fool.feature)
y_pred = kmeans(*fool.features(), plot=True, repeat=50, weight=[0.4, 0.6])