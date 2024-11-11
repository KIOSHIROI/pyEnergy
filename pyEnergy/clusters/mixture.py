import numpy as np
from pyEnergy.clusters.model import Model, compute_score
from sklearn.mixture import GaussianMixture as gmm

class Gaussian(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(gmm)

    def si(self, **params):
        '''params:'''
        data = self.fool.feature
        max_clusters = params.get("max_clusters", int(np.ceil(np.sqrt(data.shape[0]))))
        metric = params.get("metric", "euclidean")
        repeats = params.get("repeats", 50)
        weights = params.get("weights", [1, 0])

        scores, best_labels = [], []
        for n_clusters in range(2, max_clusters + 1):
            score = float('-inf')
            best_cluster_labels = None
            
            for _ in range(repeats):
                model = self.model(n_components=n_clusters, random_state=43)
                cluster_labels = model.fit_predict(data)
                new_score = compute_score(data, cluster_labels, weights, metric=metric)

                if new_score > score:
                    score = new_score
                    best_cluster_labels = cluster_labels

            scores.append(score)
            best_labels.append(best_cluster_labels)

        best_idx = np.argmax(scores)
        
        return best_labels[best_idx], scores[best_idx], best_idx + 2
    
    


