from pyEnergy.cluster import kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.drawer import draw_corr
from pyEnergy.fool import initialize_with_feature_selector, initialize
from pyEnergy.clusters.mixture import gmm_si_penal
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(
    path, method="corr",
    selection_params={"n_components": 5, "threshold": 0.6}, drought=False)
# fool = initialize(path)
# y_pred = kmeans_elbow1(*fool.features(), plot=True, repeat=50)
y_pred = gmm_si_penal(*fool.features(), plot=True, repeat=50)
fool.feature["Cluster"]=y_pred
fool.feature_backup["Cluster"]=y_pred
###* ----------model selection end----------------------
for j in range(2, 10):
    # #! reduce = "gaussian" | "moving" | "wavelet" | ""
    # reduce = "wavelet"
    sols, err = signal_composition(fool, event_param="realP_B", feature_param=fool.param_feature_dict["realP"][1], 
                                   reduce="my", reduce_params={"threshold":2, "delta": 0.1}, index=j, up_bound=None, n_color=7, color_reverse=True)
