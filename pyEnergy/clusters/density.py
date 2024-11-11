import numpy as np
from sklearn.cluster import DBSCAN as db
from sklearn.cluster import OPTICS as op
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from pyEnergy.clusters.model import Model

class DBSCAN(Model):
    def __init__(self, fool, **params):
        super().__init__(fool)
        self.use(db)
        self.y_all = None

    def plot(self):
        if self.y_all is not None:
            labels = self.y_all
            data = self.features.copy()
            data['Cluster'] = labels

            palette = sns.color_palette("hsv", len(np.unique(labels)))
            sns.pairplot(data, hue='Cluster', palette=palette, diag_kind='kde', plot_kws={'alpha': 0.5})

            plt.suptitle('DBSCAN', y=1.02)
            plt.show()
        else:
            print("y_all is None, please fit firstly.")
            

    def si(self, **params):
        eps_values = params.get("eps_range", np.arange(0.01, 1, 0.01))  # eps 值范围
        min_samples_values = params.get("min_samples_range", range(2, 11))  # min_samples 值范围
        weights = params.get("weights", [1, 0])
        metric = params.get("metric", 'euclidean')
        
        best_score = -1
        best_params = (None, None)

        data = self.fool.feature

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = self.model(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                
                # 计算轮廓系数（只考虑有效标签）
                if len(set(labels[labels!=-1])) > 1 and len(set(labels)) < len(data) - 1:  # 排除只有一个聚类或仅有噪声的情况
                    score, *_ = compute_score(data, labels, weights, metric)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        best_labels = labels  # 记录最佳标签
        mask = best_labels != -1
        self.y_all = best_labels
        self.y_pred = best_labels[mask]
        self.features = self.fool.feature.copy()
        self.fool.feature = self.fool.feature[mask].copy()
        self.fool.feature_backup = self.fool.feature_backup[mask].copy()

        return best_params, best_score

    def fit(self, **params):
        eps, min_samples = params.get("eps"), params.get("min_samples")
        if eps == None or min_samples == None:
            best_params, best_score = self.si(**params)
        else:
            weights = params.get("weights", [1, 0])
            metric = params.get("metric", 'euclidean') 
            labels = self.model(eps=eps, min_samples=min_samples).fit_predict(self.fool.feature)
            best_params, best_score = [eps, min_samples], compute_score(self.fool.feature, labels, weights, metric)
            self.y_pred = labels
        print(f"最佳参数: eps={best_params[0]}, min_samples={best_params[1]}, 轮廓系数={best_score}")

        return self.y_pred


class OPTICS(Model):
    def __init__(self, fool, **params):
        super().__init__(fool)
        self.use(op)

    def plot(self):
        # 获取聚类标签和数据
        labels = self.y_pred  # 聚类标签
        data = self.fool.feature.copy()  # 确保数据是 DataFrame
        data['Cluster'] = labels  # 将聚类标签作为新列添加到数据中

        # 使用 pairplot 展示
        palette = sns.color_palette("hsv", len(np.unique(labels)))
        sns.pairplot(data, hue='Cluster', palette=palette, markers=["o", "X", "s", "D", "P"], diag_kind='kde')

        plt.suptitle('OPTICS', y=1.02)
        plt.show()

    def si(self, **params):
        xi_values = params.get("xi_values", np.arange(0.01, 0.5, 0.05))  # xi 值范围
        min_samples_values = params.get("min_samples_values", range(2, 11))  # min_samples 值范围
        weight = params.get("weight", [1, 0])
        metric = params.get("metric", 'euclidean')
        best_score = -1
        best_params = (None, None)

        data = self.fool.feature

        for xi in xi_values:
            for min_samples in min_samples_values:
                optics = self.model(min_samples=min_samples, xi=xi)
                labels = optics.fit_predict(data)

                # 计算轮廓系数（只考虑有效标签）
                if len(set(labels)) > 1 and -1 not in labels:  # 排除只有一个聚类或仅有噪声的情况
                    score, *_ = compute_score(data, labels, weight, metric)
                    if score > best_score:
                        best_score = score
                        best_params = (xi, min_samples)
                        best_labels = labels  # 记录最佳标签

        self.y_pred = best_labels  # 保存最佳聚类标签
        return best_params, best_score

    def fit(self, **params):
        if params != None:
            best_params, best_score = params.get("xi", 0.05), params.get("min_samples", 5)
        best_params, best_score = self.si(**params)
        print(f"最佳参数: xi={best_params[0]}, min_samples={best_params[1]}, 轮廓系数={best_score}")
        return self.y_pred

def compute_score(data, labels, weights, metric):
    mask = labels != -1
    if sum(mask) < 2:
        return -1, -1, -1

    valid_labels = labels[mask]
    S_coeff = silhouette_samples(data[mask], valid_labels, metric=metric)
    mean_score = np.mean(S_coeff)
    std_score = np.std(S_coeff)

    perf_val = weights[0] * mean_score - weights[1] * std_score

    return perf_val, mean_score, std_score
