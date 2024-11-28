import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.model_selection import ParameterGrid
from pyEnergy.clusters.model import Model

class ISODATA(Model):
    def __init__(self, fool):
        super().__init__(fool)

    def fit(self, **params):
        # 如果设置了自动调参，则进行超参数搜索
        if params.get("auto_tune", False):
            best_params = self.auto_tune(params)
            print("Best Params:", best_params)
            # 更新 params 为最佳参数
            params.update(best_params)

        # 进行 ISODATA 聚类
        self.init_k = params.get("init_k", 5)  # 初始聚类数量
        self.merge_dist = params.get("merge_dist", 1.0)  # 合并距离阈值
        self.split_threshold = params.get("split_threshold", 0.7)  # 分裂阈值

        # 获取数据并确保其是 numpy 数组
        data = self.fool.feature
        if isinstance(data, pd.DataFrame):
            data = data.values

        self.y_pred = None
        self.centroids = None
        self.n_clusters = self.init_k

        # 初始聚类（例如使用 KMeans 来初始化簇中心）
        self._initialize_clusters(data)

        # 进行迭代，直到满足收敛条件
        prev_y_pred = None
        for iteration in range(params.get("max_iter", 10)):
            print(f"Iteration {iteration + 1}")

            # 根据当前簇中心进行预测（赋予每个点最近的簇）
            self.y_pred = self._assign_clusters(data)

            # 打印当前的簇信息
            print(f"Number of unique clusters: {len(np.unique(self.y_pred))}")
            print(f"Current cluster assignments: {np.unique(self.y_pred)}")

            # 如果簇的数量少于 2，跳过轮廓系数的计算
            if len(np.unique(self.y_pred)) < 2:
                print("Only one cluster, skipping silhouette score.")
            else:
                # 计算并输出当前的轮廓系数（可以用来调优或观察聚类效果）
                score = silhouette_score(data, self.y_pred)
                print(f"Silhouette score: {score}")

            # 如果簇标签没有变化，说明收敛了
            if np.array_equal(self.y_pred, prev_y_pred):
                break
            prev_y_pred = self.y_pred

            # 更新簇的中心
            self._update_centroids(data)

            # 分裂或合并簇
            self._split_or_merge_clusters(data)

        return self

    def _initialize_clusters(self, data):
        # 使用 KMeans 等方法初始化聚类中心
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.init_k, random_state=42)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_
        self.y_pred = kmeans.labels_
        self.n_clusters = self.init_k

    def _assign_clusters(self, data):
        # 计算每个数据点与簇中心的距离
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, data):
        # 更新每个簇的中心
        for i in range(self.n_clusters):
            cluster_points = data[self.y_pred == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)

    def _split_or_merge_clusters(self, data):
        # 分裂和合并簇的条件逻辑
        for i in range(self.n_clusters):
            cluster_points = data[self.y_pred == i]
            if len(cluster_points) < self.split_threshold * len(data) / self.n_clusters:
                # 执行分裂操作
                print(f"Splitting cluster {i}")
                self._split_cluster(i, cluster_points)
            else:
                # 检查簇是否需要合并
                for j in range(i + 1, self.n_clusters):
                    dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    if dist < self.merge_dist:
                        # 合并簇
                        print(f"Merging clusters {i} and {j}")
                        self._merge_clusters(i, j)

    def _split_cluster(self, cluster_index, cluster_points):
        # 进行簇的分裂操作
        new_centroid = np.mean(cluster_points, axis=0) + np.random.randn(*cluster_points.shape[1:])
        self.centroids = np.vstack([self.centroids, new_centroid])
        self.n_clusters += 1
        self.y_pred[self.y_pred == cluster_index] = self.n_clusters - 1

    def _merge_clusters(self, cluster_i, cluster_j):
        # 进行簇合并（例如，取两个簇的平均中心）
        new_centroid = np.mean([self.centroids[cluster_i], self.centroids[cluster_j]], axis=0)
        self.centroids[cluster_i] = new_centroid
        self.centroids = np.delete(self.centroids, cluster_j, axis=0)
        self.n_clusters -= 1
        self.y_pred[self.y_pred == cluster_j] = cluster_i

    def auto_tune(self, params):
        """
        自动调参方法，通过网格搜索选择最佳 init_k、merge_dist 和 split_threshold
        """
        # 定义参数搜索范围，可以根据数据分布调整范围
        param_grid = {
            'init_k': params.get("init_k_range", [2, 4, 7, 10]),
            'merge_dist': params.get("merge_dist_range", [0.5, 0.8, 1.0, 1.2]),
            'split_threshold': params.get("split_threshold_range", [0.3, 0.5, 0.7])
        }

        best_score = float('-inf')
        best_params = None
        data = self.fool.feature  # 获取特征数据

        # 确保 data 是 numpy 数组而不是 pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values

        # 遍历所有参数组合
        for param_comb in ParameterGrid(param_grid):
            # 备份原始参数，避免直接修改
            params_copy = params.copy()
            # 更新参数
            params_copy.update(param_comb)
            print(f"Trying params: {param_comb}")

            # 使用当前参数进行聚类
            self.init_k = params_copy['init_k']
            self.merge_dist = params_copy['merge_dist']
            self.split_threshold = params_copy['split_threshold']
            
            # Initialize clustering and fit
            self._initialize_clusters(data)

            # Perform the clustering
            self.y_pred = self._assign_clusters(data)
            self._update_centroids(data)
            self._split_or_merge_clusters(data)

            # If the clustering result is not empty
            if self.y_pred is not None and len(np.unique(self.y_pred)) > 1:
                # Calculate silhouette score as evaluation metric
                current_score = silhouette_score(data, self.y_pred)
                print(f"Silhouette score: {current_score}")

                # Record the best parameters and silhouette score
                if current_score > best_score:
                    best_score = current_score
                    best_params = param_comb
                    print(f"New best score: {best_score} with params: {best_params}")
            else:
                print("Skipping silhouette score calculation due to single cluster.")

        # Return the best parameters
        return best_params if best_params else params
