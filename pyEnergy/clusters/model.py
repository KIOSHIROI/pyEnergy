from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples
import numpy as np
from seaborn import pairplot
from pyEnergy.drawer import draw_silhouette_scores

class Model:
    def __init__(self, fool):
        self.fool = fool
        self.y_pred = None
        pass
    def use(self, model):
        self.model = model

    def fit(self, **params):
        self.y_pred, score, n_clusters = self.si(**params)
        print(f"best_n_clusters: {n_clusters}, score: {score}")   
        return self.y_pred
    
    def plot(self, plot=True):
            if self.y_pred is not None:
                y_pred_df = self.fool.feature.copy()
                y_pred_df['cluster'] = self.y_pred
                pairplot(y_pred_df, hue='cluster')
                plt.show()
            else:
                print("y_pred is None, please 'fit' firstly.")
    
    def si(self, **params):
        '''params:'''
        data = self.fool.feature
        max_clusters = params.get("max_clusters", int(np.ceil(np.sqrt(data.shape[0]))))
        metric = params.get("metric", "euclidean")
        repeats = params.get("repeats", 50)
        weights = params.get("weights", [1, 0])
        print(f"Max cluster:{max_clusters}\nRepeats:{repeats}\nWeights:μ{weights[0]},σ{weights[1]}")
        print("-"*15)

        scores, best_labels = [], []
        for n_clusters in range(2, max_clusters + 1):
            score = float('-inf')
            best_cluster_labels = None
            
            for _ in range(repeats):
                model = self.model(n_clusters=n_clusters, random_state=43)
                cluster_labels = model.fit_predict(data)
                new_score = compute_score(data, cluster_labels, weights, metric=metric)

                if new_score > score:
                    score = new_score
                    best_cluster_labels = cluster_labels

            scores.append(score)
            best_labels.append(best_cluster_labels)

        best_idx = np.argmax(scores)
        plot = params.get("plot", True)
        if plot:
            draw_silhouette_scores(max_clusters, scores)
        return best_labels[best_idx], scores[best_idx], best_idx + 2



def compute_score(data, labels, weights, metric):
    S_coeff = silhouette_samples(data, labels, metric=metric)
    unique_clusters = np.unique(labels)
    std_tmp = np.array([np.std(S_coeff[labels == cluster]) for cluster in unique_clusters])
    Scores_mean = np.mean(S_coeff)
    Scores_std = -np.mean(std_tmp)
    perf_val = weights[0] * Scores_mean + weights[1] * Scores_std 
    return perf_val


