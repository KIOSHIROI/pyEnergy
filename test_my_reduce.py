from pyEnergy.cluster import kmeans_elbow1
from pyEnergy.final import my_reduce, signal_composition
from pyEnergy.drawer import draw_corr, draw_signal, draw_signal_series
from pyEnergy.fool import initialize_with_feature_selector, initialize
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(
    path, method="corr",
    selection_params={"n_components": 5, "threshold": 0.3}, drought=True)
data = fool.other_event[12]
draw_signal(data, param_str="curnt_B", ax=None)
signal = my_reduce(data['curnt_B'])
draw_signal_series(signal)


