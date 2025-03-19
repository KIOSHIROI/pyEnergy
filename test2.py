import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pyEnergy.CONST as CONST
from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.two_stage import TwoStageCluster
from pyEnergy.clusters.kmeans import Kmeans
from pyEnergy.clusters.HAC import HAC
from pyEnergy.eval import evaluate_specific_output, setup_logging
import os
# 数据路径配置
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "ChangErZhai-40-139079-values 20180101-20181031.csv"))
print(f"使用数据文件: {path}")

###?---特征选择配置---
# method: 'pca' - 主成分分析 | 'corr' - 相关性分析
# threshold: 相关性阈值（用于corr方法）
# n_components: 主成分数量（用于pca方法）
OLD_FEATURES_1 = [CONST.feature_info[1], CONST.feature_info[9]] # 10 
OLD_FEATURES_11 = [CONST.feature_info[0], CONST.feature_info[1], CONST.feature_info[9]]

OLD_FEATURES_2 = [CONST.feature_info[9], CONST.feature_info[5]] # 2
OLD_FEATURES_3 = [CONST.feature_info[0], CONST.feature_info[1], # 2
                CONST.feature_info[2], CONST.feature_info[3]]

fool = Fool(path).select("selected", selected_features=OLD_FEATURES_1)

###?---创建两阶段聚类模型---
model = Kmeans(fool)
###?---聚类参数配置---
# min_samples: 每个簇的最小样本数
# max_clusters: 最大簇数
# metric: 距离度量方式
# repeats: 重复聚类次数
# plot: 是否显示聚类结果图
cluster_params = {
    'min_samples': 1,
    'max_clusters': 10,
    # 'metric': 'euclidean',
    'repeats': 10,
    'plot': False,
    'weights': [0.25, 0.75]
}
model_and_attr = "OLD_METHOD5511/"
###?---执行聚类分析---
y_pred, score, n_clusters = model.fit(**cluster_params)
# model.plot()
print(f"聚类完成: 最佳簇数={n_clusters}")

##?---负荷分解配置---
# threshold: 合并阈值
# fit: 是否进行参数拟合
composer = Composer(model.fool, y_pred, threshold=1)
composer.split_blocks()
composer.set_param('curnt_B', fit=False)
composer.set_reducer('my2')
# composer.compose(index=0)
# composer.plot()
###?---执行负荷分解---

output_prefix = f"output/{model_and_attr}/data/"
print(f"开始负荷分解，输出文件前缀: {output_prefix}")
auto_compose(composer, output_prefix, param="curnt_B")

setup_logging()# 评估特定的输出结果
output_prefix = os.path.join("output", model_and_attr)
cluster_params_string = "w10_noL"
evaluate_specific_output(output_prefix, cluster_params_string)
