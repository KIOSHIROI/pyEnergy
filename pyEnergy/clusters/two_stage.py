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
        执行两阶段聚类
        """
        min_samples = params.get("min_samples", 2)
        max_clusters = params.get("max_clusters", min(int(np.ceil(np.sqrt(self.fool.feature.shape[0]))), 20))
        metric = params.get("metric", "euclidean")
        repeats = params.get("repeats", 50)
        weights = params.get("weights", [1, 0])
        plot = params.get("plot", True)
    
        if hasattr(self.fool, 'power_feature'):
            power_feature = self.fool.power_feature
        else:
            power_feature = self.fool.feature.iloc[:, 0].values.reshape(-1, 1)
            print("missing power feature, use first feature instead.")
    
        # 其他特征
        other_features = self.fool.other_features
        power_scores, power_labels = [], []
        for n_clusters in range(2, max_clusters + 1):
            cluster_scores = []
            cluster_labels = []
            # 对功率特征进行聚类
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
        best_power_idx = np.argmax(power_scores)
        best_power_labels = power_labels[best_power_idx]
    
        # 对其他特征进行聚类
        final_scores, final_labels = [], []
        for n_clusters in range(2, max_clusters + 1):
            cluster_scores = []
            cluster_labels = []
    
            def run_single_clustering():
                model = KMeans(n_clusters=n_clusters, random_state=np.random.randint(0, 10000))
                labels = model.fit_predict(other_features)
                if min(np.bincount(labels)) >= min_samples:  
                    return labels
                return None
    
            results = Parallel(n_jobs=-1)(
                delayed(run_single_clustering)() for _ in range(repeats)
            )
    
            for labels in results:
                if labels is not None:
                    score = silhouette_score(other_features, labels, metric=metric)
                    cluster_scores.append(score)
                    cluster_labels.append(labels)
    
            if cluster_scores:
                best_idx = np.argmax(cluster_scores)
                final_scores.append(cluster_scores[best_idx])
                final_labels.append(cluster_labels[best_idx])
            else:
                final_scores.append(float('-inf'))
                final_labels.append(None)
    
        # 添加错误处理
        if all(label is None for label in final_labels):
            raise ValueError("所有聚类尝试都失败，请调整参数")

        best_final_idx = np.argmax(final_scores)
        best_final_labels = final_labels[best_final_idx]

        # 可选：结合功率特征和其他特征的聚类结果
        if best_power_labels is not None and best_final_labels is not None:
            # 这里可以添加结合两种聚类结果的逻辑
            pass

        if plot:
            plt.plot(range(2, max_clusters + 1), power_scores, marker='o', label='Power Feature')
            plt.plot(range(2, max_clusters + 1), final_scores, marker='o', label='Other Features')
            plt.title("Silhouette Score vs Number of Clusters")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")
            plt.legend()
            plt.grid(True)
            plt.show()
    
        self.y_pred = best_final_labels
        return self.y_pred, final_scores[best_final_idx], best_final_idx + 2