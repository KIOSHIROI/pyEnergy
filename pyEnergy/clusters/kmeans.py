import warnings
import os
from pyEnergy.clusters.model import Model
from sklearn.cluster import KMeans as km

# 设置环境变量以避免Windows上的内存泄漏
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=FutureWarning)
class Kmeans(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(km)
        self.model_name = "Kmeans"





