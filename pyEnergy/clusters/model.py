from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
import numpy as np
from seaborn import pairplot
from pyEnergy.drawer import draw_silhouette_scores
from sklearn.cluster import KMeans


class Model:
    def __init__(self, fool):
        self.fool = fool
        self.y_pred = None
        self.model = None  
        self.model_name = None  
    
    def string(self):
        return self.model_name

    def use(self, model):
        self.model = model

    def fit(self, **params):
        # 检查数据有效性
        if self.fool.feature.empty:
            raise ValueError("数据为空，无法进行聚类分析")
        if self.fool.feature.shape[0] < 2:
            raise ValueError("样本数量不足，至少需要2个样本")
        if self.fool.feature.shape[1] < 1:
            raise ValueError("特征数量不足，至少需要1个特征")
            
        f = self.f_selection(params.get("f", "si"))
        try:
            self.y_pred, score, n_clusters = f(**params)
            if self.y_pred is None:
                raise ValueError("聚类结果无效，可能是由于数据质量问题")
            print(f"best_n_clusters: {n_clusters}, score: {score}")
            return self.y_pred
        except Exception as e:
            print(f"聚类过程发生错误: {str(e)}")
            return None

    def plot(self, plot=True):
        if self.y_pred is not None:
            y_pred_df = self.fool.feature.copy()
            y_pred_df['cluster'] = self.y_pred
            pairplot(y_pred_df, hue='cluster')
            plt.show()
        else:
            print("y_pred is None, please 'fit' firstly.")

    def f_selection(self, f):
        functions = {
            "si": self.si,
            "db": self.db,
            "ch": self.ch
        }
        if f in functions:
            return functions[f]
        raise ValueError(f"Invalid function: {f}")


    
    def si(self, **params):
                min_samples = params.get("min_samples", 2)
                data = self.fool.feature
                max_clusters = params.get("max_clusters", min(int(np.ceil(np.sqrt(data.shape[0]))), 20))
                metric = params.get("metric", "euclidean")
                repeats = params.get("repeats", 50)
                weights = params.get("weights", [1, 0])
                plot = params.get("plot", True)
        
                assert data.shape[0] > 1, "数据样本数太少，无法聚类"
                assert len(weights) == 2, "weights 参数必须有两个值"
                assert max_clusters > 1, "max_clusters 必须大于 1"
                
                print(f"Max cluster: {max_clusters}\nRepeats: {repeats}\nWeights: μ={weights[0]}, σ={weights[1]}")
                print("-" * 15)
        
                scores, best_labels = [], []
                for n_clusters in range(2, max_clusters + 1):
                    cluster_scores = []
                    cluster_labels = []
                    
                    def run_single_clustering():
                        model = self.model(n_clusters, random_state=np.random.randint(0, 10000))
                        labels = model.fit_predict(data)
                        if min(np.bincount(labels)) >= min_samples:  
                            return labels
                        return None
                    
                    results = Parallel(n_jobs=-1)(
                        delayed(run_single_clustering)() for _ in range(repeats)
                    )
                    
                    for labels in results:
                        if labels is not None:
                            cluster_scores.append(compute_score(data, labels, weights, metric))
                            cluster_labels.append(labels)
        
                    if cluster_scores:
                        best_idx = np.argmax(cluster_scores)
                        scores.append(cluster_scores[best_idx])
                        best_labels.append(cluster_labels[best_idx])
                    else:
                        scores.append(float('-inf'))
                        best_labels.append(None)
                
                best_idx = np.argmax(scores)
                if plot:
                    draw_silhouette_scores(max_clusters, scores)
                print(f"分数:{scores[best_idx]}")
                return best_labels[best_idx], scores[best_idx], best_idx + 2


    def db(self, **params):
        min_samples = params.get("min_samples", 2)
        data = self.fool.feature
        max_clusters = params.get("max_clusters", min(int(np.ceil(np.sqrt(data.shape[0]))), 20))
        repeats = params.get("repeats", 50)
        plot = params.get("plot", True)

        print(f"Max cluster: {max_clusters}\nRepeats: {repeats}")
        print("-" * 15)

        scores, best_labels = [], []
        
        for n_clusters in range(2, max_clusters + 1):
            best_db = float('inf')
            best_cluster_labels = None

            def run_single_clustering():
                model = self.model(n_clusters=n_clusters)
                labels = model.fit_predict(data)
                unique_labels, counts = np.unique(labels, return_counts=True)
                if min(counts) < min_samples:  
                    return None, None
                db_score = davies_bouldin_score(data, labels)
                return labels, db_score
            
            results = Parallel(n_jobs=-1)(
                delayed(run_single_clustering)() for _ in range(repeats)
            )
            
            for labels, db_score in results:
                if labels is not None and db_score < best_db:
                    best_db = db_score
                    best_cluster_labels = labels
            
            scores.append(best_db if best_cluster_labels is not None else float('inf'))
            best_labels.append(best_cluster_labels)
        
        best_idx = np.argmin(scores)
        if plot:
            plt.plot(range(2, max_clusters + 1), scores, marker='o')
            plt.title("DB Index vs Number of Clusters")
            plt.xlabel("Number of Clusters")
            plt.ylabel("DB Index")
            plt.grid(True)
            plt.show()
        print(f"分数:{scores[best_idx]}")    
        return best_labels[best_idx], scores[best_idx], best_idx + 2
    from sklearn.metrics import calinski_harabasz_score

    def ch(self, **params):
        min_samples = params.get("min_samples", 2)
        data = self.fool.feature
        max_clusters = params.get("max_clusters", min(int(np.ceil(np.sqrt(data.shape[0]))), 20))
        repeats = params.get("repeats", 50)
        plot = params.get("plot", True)

        print(f"Max cluster: {max_clusters}\nRepeats: {repeats}")
        print("-" * 15)

        scores, best_labels = [], []

        for n_clusters in range(2, max_clusters + 1):
            best_ch = float('-inf')
            best_cluster_labels = None

            def run_single_clustering():
                model = self.model(n_clusters)
                labels = model.fit_predict(data)
                unique_labels, counts = np.unique(labels, return_counts=True)
                if min(counts) < min_samples: 
                    return None, None
                ch_score = calinski_harabasz_score(data, labels)
                return labels, ch_score

            results = Parallel(n_jobs=-1)(
                delayed(run_single_clustering)() for _ in range(repeats)
            )

            for labels, ch_score in results:
                if labels is not None and ch_score > best_ch:
                    best_ch = ch_score
                    best_cluster_labels = labels

            scores.append(best_ch if best_cluster_labels is not None else float('-inf'))
            best_labels.append(best_cluster_labels)

        best_idx = np.argmax(scores)
        if plot:
            plt.plot(range(2, max_clusters + 1), scores, marker='o')
            plt.title("CH Index vs Number of Clusters")
            plt.xlabel("Number of Clusters")
            plt.ylabel("CH Index")
            plt.grid(True)
            plt.show()
        print(f"分数:{scores[best_idx]}")
        return best_labels[best_idx], scores[best_idx], best_idx + 2


def compute_score(data, labels, weights, metric):
    S_coeff = silhouette_samples(data, labels, metric=metric)
    unique_clusters = np.unique(labels)
    std_tmp = np.array([np.std(S_coeff[labels == cluster]) for cluster in unique_clusters])
    Scores_mean = np.mean(S_coeff)
    Scores_std = -np.mean(std_tmp)
    perf_val = weights[0] * Scores_mean + weights[1] * Scores_std 
    return perf_val