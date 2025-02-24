import numpy as np
import logging
from pyEnergy.fool import Fool
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.clusters.kmeans import Kmeans
import pyEnergy.CONST as CONST

class TwoStageCluster:
    def __init__(self, fool, method='hac', linkage='ward'):
        """
        初始化双阶段聚类模型
        
        参数:
            fool: Fool对象，包含特征数据
            method: str, 聚类方法 ('hac', 'kmeans', 'gaussian')
            linkage: str, HAC方法的连接方式
        """
        self.fool = fool
        self.method = method.lower()
        self.linkage = linkage
        self.y_pred_stage1 = None
        self.y_pred = None
        self.cluster_mapping = {}
        
        # 选择聚类模型
        if self.method == 'hac':
            self.model_class = HAC
        elif self.method == 'kmeans':
            self.model_class = Kmeans
        elif self.method == 'gaussian':
            self.model_class = Gaussian
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
    
    def _perform_first_stage(self, **params):
        """执行第一阶段聚类（使用功率特征）"""
        logging.info("开始第一阶段聚类（功率特征）...")
        self.fool = self.fool.select("selected", selected_features=CONST.REAL_POWER_FEATURES)
        model = self.model_class(self.fool)
        self.y_pred_stage1 = model.fit(**params)
        self.stage1_model = model
        return self.y_pred_stage1
    
    def _perform_second_stage(self, non_power_features, **params):
        """执行第二阶段聚类（使用非功率特征）"""
        logging.info("开始第二阶段聚类（非功率特征）...")
        
        try:
            fool_stage2 = Fool(self.fool.csv).select(
                method='selected', 
                selected_features=non_power_features
            ).select(method='pca', n_components=3)
            
            if fool_stage2.feature.empty:
                raise ValueError("非功率特征数据为空")
            
            final_clusters = []
            self.cluster_mapping = {}
            start_label = 0
            
            # 对第一阶段的每个簇进行二次聚类
            for cluster_id in np.unique(self.y_pred_stage1):
                logging.info(f"处理第一阶段簇 {cluster_id}...")
                cluster_mask = self.y_pred_stage1 == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size < 2:
                    logging.info(f"簇 {cluster_id} 样本数量不足（{cluster_size}），跳过二次聚类")
                    final_clusters.extend([start_label] * cluster_size)
                    self.cluster_mapping[cluster_id] = [start_label]
                    start_label += 1
                    continue
                
                try:
                    # 创建子簇的特征数据
                    sub_fool = Fool(self.fool.csv)
                    sub_fool.feature = fool_stage2.feature[cluster_mask]
                    
                    if sub_fool.feature.empty:
                        raise ValueError(f"簇 {cluster_id} 的特征数据为空")
                    
                    # 对子簇进行聚类
                    sub_model = self.model_class(sub_fool)
                    sub_clusters = sub_model.fit(**params)
                    
                    if sub_clusters is None:
                        raise ValueError(f"簇 {cluster_id} 的聚类失败")
                    
                    # 更新簇标签
                    n_subclusters = len(np.unique(sub_clusters))
                    logging.info(f"簇 {cluster_id} 成功分为 {n_subclusters} 个子簇")
                    sub_labels = sub_clusters + start_label
                    final_clusters.extend(sub_labels)
                    self.cluster_mapping[cluster_id] = list(range(start_label, start_label + n_subclusters))
                    start_label += n_subclusters
                    
                except Exception as e:
                    logging.warning(f"处理簇 {cluster_id} 时发生错误：{str(e)}")
                    final_clusters.extend([start_label] * cluster_size)
                    self.cluster_mapping[cluster_id] = [start_label]
                    start_label += 1
            
            self.y_pred = np.array(final_clusters)
            logging.info(f"聚类完成：第一阶段{len(np.unique(self.y_pred_stage1))}个簇，最终{len(np.unique(self.y_pred))}个簇")
            logging.info(f"簇映射关系: {self.cluster_mapping}")
            
        except Exception as e:
            logging.error(f"二次聚类过程失败：{str(e)}")
            logging.info("将使用第一阶段的聚类结果")
            self.y_pred = self.y_pred_stage1
        
        return self.y_pred
    
    def fit(self, **params):
        """执行完整的双阶段聚类过程
        
        参数:
            **params: 聚类参数，包括：
                repeats: int, 重复聚类次数
                weights: list, 评分权重
                其他特定聚类算法的参数
        
        返回:
            y_pred: np.array, 最终的聚类标签
            score: float, 聚类评分
            n_clusters: int, 最终簇的数量
        """
        # 第一阶段聚类
        self._perform_first_stage(**params)
        
        # 获取非功率特征
        non_power_features = [f for f in self.fool.feature_backup.columns 
                            if f not in CONST.REAL_POWER_FEATURES]
        
        if not non_power_features:
            logging.warning("未找到非功率特征，将跳过二次聚类")
            self.y_pred = self.y_pred_stage1
        else:
            # 第二阶段聚类
            self._perform_second_stage(non_power_features, **params)
        
        # 计算聚类评分
        score = self.stage1_model.score if hasattr(self.stage1_model, 'score') else 0
        n_clusters = len(np.unique(self.y_pred))
        
        return self.y_pred, score, n_clusters
    
    def get_cluster_mapping(self):
        """获取簇映射关系"""
        return self.cluster_mapping