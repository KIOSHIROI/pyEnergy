from pyEnergy.cluster import kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.drawer import draw_corr
from pyEnergy.fool import initialize_with_feature_selector, initialize
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(
    path, method="corr",
    selection_params={"n_components": 5, "threshold": 0.3}, drought=True)
# fool = initialize(path)
y_pred = kmeans_elbow1(*fool.features(), plot=True, repeat=50)