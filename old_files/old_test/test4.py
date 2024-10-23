from pyEnergy.cluster import kmeans, kmeans_elbow1
from pyEnergy.final import reduce_signal
from test5 import signal_composition_opt1
from pyEnergy.fool import initialize
import pyEnergy.drawer as drawer
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
path = "data/WuFanZhuang-45-139040-values 20181001-20190214.csv"
fool = initialize(path, normal=False)
feature1 = fool.feature.copy()
fool.feature_selection(method="corr", selection_params={"threshold": 0.7})
# fool.feature_selection(method="filter", selection_params={"n_components": 2, "threshold": 0.5})

y_pred = kmeans_elbow1(*fool.features(), plot=False, repeat=10)

feature1["Cluster"] = y_pred

phaseB_perCluster = feature1.groupby('Cluster')["ave. phase B current(ss)"]
realP_perCluster = feature1.groupby('Cluster')["ave. real power(ss)"]
print(f"realP_perCluster:\n{realP_perCluster}")
print(f"phaseB_perCluster:\n{phaseB_perCluster}")


other_events = fool.other_event

print(f"other events num: {len(other_events)}")
event = other_events[8]
singal = event["curnt_B"]
# singal1, singal2 = reduce_signal(singal)
    
#     reconstructed_signal = reconstruct_time_series(sols, realP_perClster)
drawer.draw_signal(event, 'curnt_B') 
# event["curnt_B_reduced1"] = singal1
# event["curnt_B_reduced2"] = singal2

# drawer.draw_signal(event, "curnt_B_reduced1")
# drawer.draw_signal(event, "curnt_B_reduced2")
sols = signal_composition_opt1(phaseB_perCluster, singal)
print(sols)
