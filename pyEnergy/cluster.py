import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pyEnergy.drawer import draw_silhouette_scores

# 
def compute_silhouette_scores(data, max_clusters, repeat, metric='euclidean', penal=0): 
    # 直接使用si平均值作为分数，
    scores = []
    for n_clusters in range(2, max_clusters + 1): # 根据2-max_cluster作为聚类数训练，记录各聚类分数
        kmeans = KMeans(n_clusters=n_clusters, random_state=43, n_init="auto")
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels, metric=metric) - penal*n_clusters/len(data)
        for _ in range(repeat):
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            cluster_labels = kmeans.fit_predict(data)
            new_score = silhouette_score(data, cluster_labels, metric=metric) - penal*n_clusters/len(data)

            if new_score > score:
                score = new_score
        scores.append(score)
    return scores

def kmeans_elbow_core1(selected_features, repeat, plot, normalize=False, max_clusters=None, metric='euclidean',penal=0):
    # 标准化数据
    if normalize:
        max_val = np.max(selected_features, axis=0)
        min_val = np.min(selected_features, axis=0)
        selected_features = (selected_features - min_val) / (max_val - min_val)

    if max_clusters is None:
        max_clusters = int(np.ceil(np.sqrt(selected_features.shape[0])))

    silhouette_scores = []
    best_score = -1

    silhouette_scores = compute_silhouette_scores(selected_features, max_clusters, repeat=repeat, metric=metric, penal=penal)
    if plot:
        draw_silhouette_scores(max_clusters, silhouette_scores)

    best_score = 0
    best_cluster = 0
    for n_clusters, score in enumerate(silhouette_scores, start=2):
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_cluster = n_clusters

    return  best_cluster, best_score


def kmeans_elbow1(feature, feature_info, repeat=5, plot=False, metric='euclidean',penal=0):
    selected_features = feature[feature_info]
    n_clusters, score, *_ = kmeans_elbow_core1(selected_features, repeat, plot=plot,metric=metric,penal=penal)
    print(f"n_clusters: {n_clusters}, score: {score}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init="auto")
    y_pred = kmeans.fit_predict(selected_features)
    if plot:
        selected_features['cluster'] = y_pred
        sns.pairplot(selected_features, hue='cluster')
        plt.show()

    return y_pred


# test start ----------
def compute_score(data, labels, weights, metric): # 根据标准差对分数进行加权
    S_coeff = silhouette_samples(data, labels, metric=metric)
    unique_clusters = np.unique(labels)
    std_tmp = np.array([np.std(S_coeff[labels == cluster]) for cluster in unique_clusters])
    Scores_mean = np.mean(S_coeff)
    Scores_std = -np.mean(std_tmp)  # Minus to indicate maximization
    perf_val = weights[0] * Scores_mean + weights[1] * Scores_std
    return perf_val

def kmeans_elbow_core(data, max_clusters, repeats, weights, plot=True, metric='euclidean', n_init="auto"):
    scores = []
    best_labels = []
    
    for n_clusters in range(2, max_clusters + 1):
        score = float('-inf')  # Initialize score to negative infinity
        best_cluster_labels = None
        
        for _ in range(repeats):
            kmeans = KMeans(n_clusters=n_clusters, random_state=43, n_init=n_init)  # Set n_init=1 inside the loop
            cluster_labels = kmeans.fit_predict(data)
            new_score = compute_score(data, cluster_labels, weights, metric=metric)

            if new_score > score:
                score = new_score
                best_cluster_labels = cluster_labels

        scores.append(score)
        best_labels.append(best_cluster_labels)

    best_idx = np.argmax(scores)  # Get the index of the best score
    best_n_cluster = best_idx + 2
    best_labels = best_labels[best_idx]
    best_score = scores[best_idx]
    if plot:
        draw_silhouette_scores(max_clusters, scores)
    
    return best_labels, best_score, best_n_cluster


def kmeans_elbow(selected_features, max_clusters=None, repeats=5, weights=[0.5, 0.5], plot=True, metric='euclidean'):
    if max_clusters is None:
        max_clusters = int(np.ceil(np.sqrt(selected_features.shape[0])))

    return  kmeans_elbow_core(selected_features, max_clusters, repeats, weights, plot, metric)


def kmeans(feature, feature_info, max_clusters=None, repeat=5, plot=True, weight=[0.5, 0.5], metric='euclidean'):
    selected_features = feature[feature_info]
    _, score, n_clusters = kmeans_elbow(selected_features, max_clusters=max_clusters, repeats=repeat, plot=plot, weights=weight, metric=metric)
    print(f"best_n_clusters: {n_clusters}, score: {score}")   
    kmeans = KMeans(n_clusters=n_clusters, random_state=43)
    y_pred = kmeans.fit_predict(selected_features)
    if plot:
        selected_features['cluster'] = y_pred
        sns.pairplot(selected_features, hue='cluster')
        plt.show()

    return y_pred

# test end------------------
