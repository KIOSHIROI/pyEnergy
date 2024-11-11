import warnings
from pyEnergy.clusters.model import Model
from sklearn.cluster import KMeans as km

warnings.filterwarnings("ignore", category=FutureWarning)
class Kmeans(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(km)





