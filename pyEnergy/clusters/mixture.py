import numpy as np
from pyEnergy.clusters.model import Model, compute_score
from sklearn.mixture import GaussianMixture as gmm

class Gaussian(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(gmm)



