import numpy as np
import os
import datetime
import logging
from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.clusters.kmeans import Kmeans 
import pyEnergy.CONST as CONST
import pyEnergy.compute as cm

# 配置日志记录
log_dir = os.path.join('output', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
path = os.path.join("data", "ChangErZhai-40-139079-values 20180101-20181031.csv")
print("使用数据文件:", path)

# 第一阶段：使用功率特征进行聚类
print("\n第一阶段：使用功率特征聚类")

import pickle

fool = pickle.load(open(os.path.join('fool', 'REAL_POWER_FEATURE_FOOL'), 'rb'))
model = HAC(fool)
y_pred_stage1 = model.fit(repeats=20, weights=[0.5, 0.5])
model.plot(plot=False)

# 第二阶段：对每个簇使用非功率特征进行二次聚类
print("\n第二阶段：使用非功率特征进行二次聚类")
# 获取非功率特征

non_power_features = [f for f in fool.feature_backup.columns if f not in CONST.REAL_POWER_FEATURES]
if not non_power_features:
    print("警告：未找到非功率特征，将跳过二次聚类")
    y_pred = y_pred_stage1
else:
    try:
        fool_stage2 = Fool(path).select(method='selected', selected_features=non_power_features).select(method='pca', n_components=3)
        if fool_stage2.feature.empty:
            raise ValueError("非功率特征数据为空")

        # 存储最终的聚类结果
        final_clusters = []
        cluster_mapping = {}
        start_label = 0

        # 对第一阶段的每个簇进行二次聚类
        for cluster_id in np.unique(y_pred_stage1):
            print(f"\n处理第一阶段簇 {cluster_id}...")
            # 获取当前簇的样本索引
            cluster_mask = y_pred_stage1 == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < 2:  # 如果簇中样本数小于2，跳过二次聚类
                print(f"簇 {cluster_id} 样本数量不足（{cluster_size}），跳过二次聚类")
                final_clusters.extend([start_label] * cluster_size)
                cluster_mapping[cluster_id] = [start_label]
                start_label += 1
                continue
            
            try:
                # 创建子簇的特征数据
                sub_fool = Fool(path)
                sub_fool.feature = fool_stage2.feature[cluster_mask]
                
                if sub_fool.feature.empty:
                    raise ValueError(f"簇 {cluster_id} 的特征数据为空")
                
                # 对子簇进行聚类
                sub_model = HAC(sub_fool)
                sub_clusters = sub_model.fit(repeats=10, weights=[0.5, 0.5])
                sub_model.plot(plot=False)
                
                if sub_clusters is None:
                    raise ValueError(f"簇 {cluster_id} 的聚类失败")
                
                # 更新簇标签
                n_subclusters = len(np.unique(sub_clusters))
                print(f"簇 {cluster_id} 成功分为 {n_subclusters} 个子簇")
                sub_labels = sub_clusters + start_label
                final_clusters.extend(sub_labels)
                cluster_mapping[cluster_id] = list(range(start_label, start_label + n_subclusters))
                start_label += n_subclusters
                
            except Exception as e:
                print(f"警告：处理簇 {cluster_id} 时发生错误：{str(e)}")
                # 发生错误时，将整个簇作为一个类别
                final_clusters.extend([start_label] * cluster_size)
                cluster_mapping[cluster_id] = [start_label]
                start_label += 1

        # 转换为numpy数组
        y_pred = np.array(final_clusters)
        print(f"\n聚类完成：第一阶段{len(np.unique(y_pred_stage1))}个簇，最终{len(np.unique(y_pred))}个簇")
        print("簇映射关系:", cluster_mapping)
        
    except Exception as e:
        print(f"\n错误：二次聚类过程失败：{str(e)}")
        print("将使用第一阶段的聚类结果")
        y_pred = y_pred_stage1

# 执行负荷分解
print("\n开始执行负荷分解...")
composer = Composer(model.fool, y_pred, threshold=1)
composer.set_param('curnt_B', fit=False)
composer.set_reducer('my2')

# 设置输出文件前缀并执行分解
output_prefix = os.path.join("output", f"{model.string()}_two_stage", "data", f"{model.string()}_two_stage_clusters{len(np.unique(y_pred))}")
print(f"输出文件前缀: {output_prefix}")

# 创建输出目录
image_dir = os.path.join("output", f"{model.string()}_two_stage", 'images')
os.makedirs(image_dir, exist_ok=True)

# 执行分解并保存结果
auto_compose(composer, output_prefix)
composer.plot(plot=False, save_path=image_dir)

# 导入评估模块
from tmp import evaluate_specific_output

# 准备聚类参数信息
cluster_info = {
    'stage1_clusters': len(np.unique(y_pred_stage1)),
    'final_clusters': len(np.unique(y_pred)),
    'params': 'repeats: 20, weights: [0.5, 0.5]'
}

# 评估分解结果
cluster_params = f"第一阶段簇数: {cluster_info['stage1_clusters']}, 最终簇数: {cluster_info['final_clusters']}, {cluster_info['params']}"
evaluate_specific_output(output_prefix, cluster_params)

