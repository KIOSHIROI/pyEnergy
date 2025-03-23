import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pyEnergy.CONST as CONST
import pyEnergy.drawer as dw
from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.two_stage import TwoStageCluster
from pyEnergy.clusters.kmeans import Kmeans
from pyEnergy.clusters.HAC import HAC
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.eval import evaluate_specific_output, setup_logging, split_events_by_cluster, save_validation_events, find_validation_set_events, validate_pump_events, interpret_validation_results, visualize_well_operations
from pyEnergy.reconstruct import reconstruct_all_signals
from pyEnergy.drawer import draw_signal
import os

import pandas as pd
# 数据路径配置
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "ChangErZhai-40-139079-values 20180101-20181031.csv"))
print(f"使用数据文件: {path}")

###?---特征选择配置---
# method: 'pca' - 主成分分析 | 'corr' - 相关性分析
# threshold: 相关性阈值（用于corr方法）
# n_components: 主成分数量（用于pca方法）
OLD_FEATURES_1 = [CONST.feature_info[1], CONST.feature_info[9]] # 10 
OLD_FEATURES_11 = [CONST.feature_info[0], CONST.feature_info[1], CONST.feature_info[9]]
ONE_FEATURES = [CONST.feature_info[1]]
OLD_FEATURES_2 = [CONST.feature_info[9], CONST.feature_info[5]] # 2
OLD_FEATURES_3 = [CONST.feature_info[0], CONST.feature_info[1], # 2
                CONST.feature_info[2], CONST.feature_info[3]]
CRT_FEATURES = [CONST.feature_info[9], CONST.feature_info[5]]

fool = Fool(path).select("selected", selected_features=CRT_FEATURES)

###?---创建两阶段聚类模型---
model = Gaussian(fool)
###?---聚类参数配置---
# min_samples: 每个簇的最小样本数
# max_clusters: 最大簇数
# metric: 距离度量方式
# repeats: 重复聚类次数
# plot: 是否显示聚类结果图
cluster_params = {
    'min_samples': 1,
    'max_clusters': 8,
    # 'metric': 'euclidean',
    'repeats': 50,
    'plot': False,
    'weights': [1, 0]
}
model_and_attr = "CRT1_10/"
###?---执行聚类分析---
y_pred, score, n_clusters = model.fit(**cluster_params)
# model.plot()
print(f"聚类完成: 最佳簇数={n_clusters}")


