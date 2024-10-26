from pyEnergy.cluster import kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.fool import initialize_with_feature_selector
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(path, method="corr", 
                                        selection_params={"n_components": 3, "threshold": 0.65})
###* ----------feature selection end--------------------
###* ----------model selection start--------------------
y_pred = kmeans_elbow1(*fool.features(), plot=False, repeat=50)
fool.feature["Cluster"]=y_pred
fool.feature_backup["Cluster"]=y_pred

###* ----------model selection end----------------------
for j in range(1, 10):
    #! reduce = "gaussian" | "moving" | "wavelet" | ""
    reduce = "wavelet"
    sols, err = signal_composition(fool, event_param="realP_B", feature_param=fool.param_feature_dict["realP"][1], 
                                   reduce="my", reduce_params={"sigma": 0.3}, index=j, up_bound=None, n_color=7, color_reverse=True)
