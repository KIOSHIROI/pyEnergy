from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.clusters.kmeans import Kmeans 

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
print(path)
for encoding_dim in range(5, 30, 5):
    ###?---you can change 0 start: method and threshold/n_components | pca->n_components, corr->threshold---
    fool = Fool(path).select(method='autoencoder', threshold=0.3, n_components=8, encoding_dim=30, epochs=500)
    ###?---you can change 0 end---

    ###?---you can change 1 start: model type---
    model = Kmeans(fool)
    ##?---you can change 1 end---

    for function in ['si', 'db', 'ch']:
        ###?---you can change 2 start: model.fit params---
        y_pred = model.fit(f=function, plot=False)
        ###?---you can change 2 end---
        for fit in [True, False]:
            ###?---remember to change output prefix---
            output_prefix = "output/" + "Kmeans/" + f"autoencoder{encoding_dim}/"
            f"Kmeans_{function}_autoencoder{encoding_dim}/" 
            
            output_prefix += "" if fit else "_fitFalse"
            print(f"---function:{function}--fit:{fit}-")

            composer = Composer(model.fool, y_pred, threshold=1).set_param('curnt_B', fit=fit).set_reducer('my2')
            if composer.skip == True:
                print(f"skip {output_prefix}\n")
                print('-'*10)
                continue
            
            auto_compose(composer, output_prefix)