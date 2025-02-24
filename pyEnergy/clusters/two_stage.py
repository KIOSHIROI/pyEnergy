from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
from pyEnergy.clusters.model import Model

class TwoStageKMeans(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(KMeans)
        
    def fit(self, **params):
        """
        执行两阶段聚类：
        1. 首先对功率特征进行聚类
        2. 对每个功率聚类结果内部使用非功率特征进行二次聚类
        """
        min_samples = params.get("min_samples", 2)
        max_clusters = params.get("max_clusters", min(int(np.ceil(np.sqrt(self.fool.feature.shape[0]))), 20))
        metric = params.get("metric", "euclidean")
        repeats = params.get("repeats", 50)
        plot = params.get("plot", True)
    
        if hasattr(self.fool, 'power_feature'):
            power_feature = self.fool.power_feature
        else:
            power_feature = self.fool.feature.iloc[:, 0].values.reshape(-1, 1)
            print("missing power feature, use first feature instead.")
    
        # 其他特征
        other_features = self.fool.other_features
        
        # 第一阶段：对功率特征进行聚类
        power_scores, power_labels = [], []
        for n_clusters in range(2, max_clusters + 1):
            cluster_scores = []
            cluster_labels = []
            
            def run_single_clustering():
                model = KMeans(n_clusters=n_clusters, random_state=np.random.randint(0, 10000))
                labels = model.fit_predict(power_feature)
                if min(np.bincount(labels)) >= min_samples: 
                    return labels
                return None
    
            results = Parallel(n_jobs=-1)(
                delayed(run_single_clustering)() for _ in range(repeats)
            )
    
            for labels in results:
                if labels is not None:
                    score = silhouette_score(power_feature, labels, metric=metric)
                    cluster_scores.append(score)
                    cluster_labels.append(labels)
    
            if cluster_scores:
                best_idx = np.argmax(cluster_scores)
                power_scores.append(cluster_scores[best_idx])
                power_labels.append(cluster_labels[best_idx])
            else:
                power_scores.append(float('-inf'))
                power_labels.append(None)
        
        # 选择最佳的功率特征聚类结果
        best_power_idx = np.argmax(power_scores)
        best_power_labels = power_labels[best_power_idx]
        if best_power_labels is None:
            raise ValueError("功率特征聚类失败，请调整参数")
        
        # 第二阶段：对每个功率聚类内部进行非功率特征聚类
        unique_clusters = np.unique(best_power_labels)
        final_labels = np.zeros_like(best_power_labels)
        current_label = 0
        
        for cluster in unique_clusters:
            # 获取当前功率聚类的样本索引
            cluster_mask = best_power_labels == cluster
            cluster_features = other_features[cluster_mask]
            
            # 对当前功率聚类内的样本进行二次聚类
            best_score = float('-inf')
            best_sub_labels = None
            
            # 尝试不同的子类数量（包括1，表示不进行二次聚类）
            max_sub_clusters = min(max_clusters, len(cluster_features))
            for n_sub_clusters in range(1, max_sub_clusters + 1):
                if n_sub_clusters == 1:
                    # 不进行二次聚类，直接使用相同标签
                    sub_labels = np.zeros(len(cluster_features))
                    score = 1.0  # 当只有一个类时，设置一个较高的得分
                else:
                    # 执行二次聚类
                    model = KMeans(n_clusters=n_sub_clusters, random_state=np.random.randint(0, 10000))
                    sub_labels = model.fit_predict(cluster_features)
                    if min(np.bincount(sub_labels)) >= min_samples:
                        score = silhouette_score(cluster_features, sub_labels, metric=metric)
                    else:
                        continue
                
                if score > best_score:
                    best_score = score
                    best_sub_labels = sub_labels
            
            if best_sub_labels is None:
                # 如果二次聚类失败，将该功率聚类视为一个整体
                best_sub_labels = np.zeros(len(cluster_features))
            
            # 更新最终标签
            for sub_label in np.unique(best_sub_labels):
                mask = cluster_mask.copy()
                mask[cluster_mask] = best_sub_labels == sub_label
                final_labels[mask] = current_label
                current_label += 1
        
        if plot:
            plt.plot(range(2, max_clusters + 1), power_scores, marker='o', label='Power Feature')
            plt.title("Power Feature Clustering Score vs Number of Clusters")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        self.y_pred = final_labels
        return self.y_pred, power_scores[best_power_idx], len(unique_clusters)