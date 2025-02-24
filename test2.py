from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.two_stage import TwoStageKMeans

# 数据路径配置
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
print(f"使用数据文件: {path}")

###?---特征选择配置---
# method: 'pca' - 主成分分析 | 'corr' - 相关性分析
# threshold: 相关性阈值（用于corr方法）
# n_components: 主成分数量（用于pca方法）
fool = Fool(path).select(method='pca', threshold=0.3, n_components=6)

###?---创建两阶段聚类模型---
model = TwoStageKMeans(fool)

###?---聚类参数配置---
# min_samples: 每个簇的最小样本数
# max_clusters: 最大簇数
# metric: 距离度量方式
# repeats: 重复聚类次数
# plot: 是否显示聚类结果图
cluster_params = {
    'min_samples': 2,
    'max_clusters': 15,
    'metric': 'euclidean',
    'repeats': 50,
    'plot': True
}

###?---执行聚类分析---
print("开始执行两阶段聚类分析...")
y_pred, score, n_clusters = model.fit(**cluster_params)
print(f"聚类完成: 最佳簇数={n_clusters}, 得分={score:.4f}")

###?---负荷分解配置---
# threshold: 合并阈值
# fit: 是否进行参数拟合
composer = Composer(model.fool, y_pred, threshold=1)
composer.set_param('curnt_B', fit=True)
composer.set_reducer('my2')

###?---执行负荷分解---
output_prefix = f"output/two_stage_pca6_clusters{n_clusters}"
print(f"开始负荷分解，输出文件前缀: {output_prefix}")
auto_compose(composer, output_prefix)