import numpy as np
from pyEnergy.clusters.density import DBSCAN, OPTICS
from pyEnergy.clusters.kmeans import Kmeans 
from pyEnergy.composition.composition import Composer
from pyEnergy.fool import Fool

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
print(path)
fool = Fool(path).select(method='pca', threshold=0.3, n_components=3)
print("fool init.")
###! DBSCAN
# model = DBSCAN(fool)
# y_pred = model.fit(eps_range=np.arange(0.7,0.8, 0.001), min_samples_range=range(1,3))
###! HAC
from pyEnergy.clusters.HAC import HAC
model = HAC(fool)
y_pred = model.fit(plot=False)
# model.plot()
# print(y_pred)
composer = Composer(model.fool, y_pred, threshold=1).set_param('realP_B')
# print('composer init.')
composer.set_reducer('my2')
error  = []
for i in range(0,len(fool.other_event)):
    _, err = composer.compose(index=i)
    error.append(err)
    # composer.plot()

print(np.mean(error))

