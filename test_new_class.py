import pickle
from pyEnergy.clusters.kmeans import Kmeans
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.clusters.density import DBSCAN
from pyEnergy.fool import initialize_with_feature_selector

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"

fool = initialize_with_feature_selector(path, method="pca", selection_params={"n_components": 4, "threshold": 0.6})

# with open('fool.pkl', 'rb') as f:
#     fool = pickle.load(f)

# md = Kmeans(fool)
# md = Gaussian(fool)
md = DBSCAN(fool)
md.fit()
md.plot()
