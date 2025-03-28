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

##?---负荷分解配置---
# threshold: 合并阈值
# fit: 是否进行参数拟合
composer = Composer(model.fool, y_pred, threshold=1)
composer.set_param('curnt_B', fit=False)
composer.set_reducer('my2')
# composer.compose(index=0)
# composer.plot()
###?---执行负荷分解---

output_prefix = f"output/{model_and_attr}/data/"
print(f"开始负荷分解，输出文件前缀: {output_prefix}")
auto_compose(composer, output_prefix, param="realP_B")

output_dir = f'output/{model_and_attr}/signals/'
split_events_by_cluster(f'output/{model_and_attr}/data/_events.csv', output_dir)

# 可视化各机井类型的运行情况
fig = visualize_well_operations(output_dir)
fig.savefig(f'output/{model_and_attr}/well_operations.png')

# ###?---验证集评估---
# 读取验证集数据
validation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "Hengyuan1-301-values20180101-20181031.csv"))
print(f"\n使用验证集数据文件: {validation_path}")

# 获取验证集的启停时间
validation_events = find_validation_set_events(validation_path)
print(f"找到 {len(validation_events)} 个验证事件")
save_validation_events(validation_events, f"output/{model_and_attr}/data/validation_events.csv")

# 验证分解结果
from validate_signals import validate_signal_events, print_validation_results

# 验证signals文件夹下的分解结果
validation_events_path = f"output/{model_and_attr}/data/validation_events.csv"
signals_dir = f"output/{model_and_attr}/signals/"
accuracies = validate_signal_events(validation_events_path, signals_dir)

# 打印验证结果
print_validation_results(accuracies)

# 信号重建逻辑
param_per_c = composer.param_per_c
reconstruct_all_signals(param_per_c, model_and_attr)

# 读取series目录下的所有聚类时间序列数据
series_data = []
series_labels = []
for i in range(n_clusters):
    series_path = f"output/{model_and_attr}/series/series_cluster_{i}.csv"
    df = pd.read_csv(series_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    series_data.append(df)
    series_labels.append(f"Cluster {i}")


# 绘制所有聚类的时间序列和验证集数据
dw.plot_signal(series_data, labels=series_labels)
plt.show()