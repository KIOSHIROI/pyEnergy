from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.clusters.kmeans import Kmeans 

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
print(path)

###?---you can change 0 start: method and threshold/n_components | pca->n_components, corr->threshold---
fool = Fool(path).select(method='pca', threshold=0.3, n_components=6)
###?---you can change 0 end---

###?---you can change 1 start: model type---
model = HAC(fool)
###?---you can change 1 end---

for function in ['si', 'db', 'ch']:
    for fit in [True, False]:
        print(f"---function:{function}--fit:{fit}-")
        ###?---you can change 2 start: model.fit params---
        y_pred = model.fit(f=function, plot=False)
        ###?---you can change 2 end---
        composer = Composer(model.fool, y_pred, threshold=1).set_param('curnt_B', fit=fit).set_reducer('my2')
        ###?---remember to change output prefix---
        output_prefix = f"output/HAC_{function}_pca6" if fit else f"output/HAC_{function}_pca6"
        auto_compose(composer, output_prefix, end_idx=2)