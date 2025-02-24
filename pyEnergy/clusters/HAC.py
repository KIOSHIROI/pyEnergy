from sklearn.cluster import AgglomerativeClustering
from pyEnergy.clusters.model import Model


class HAC(Model):
    def __init__(self, fool, linkage="ward"):
        super().__init__(fool)
        self.use(linkage)
        self.model_name = "HAC"
        
    def use(self, linkage="ward", f="si"):
        self.linkage = linkage
        self.model = lambda n_clusters, **kwargs: AgglomerativeClustering(n_clusters, linkage=self.linkage)
