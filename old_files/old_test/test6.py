from pyEnergy.cluster import kmeans, kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.fool import initialize
from pyEnergy.drawer import draw_parallel_coordinates

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
### feature selection start-------------------
fool = initialize(path, normal=False)
feature1 = fool.feature[:].copy()
fool.feature_selection(method="pca", selection_params={"threshold": 0.7})
### feature selection end-----model selection start-------------------
y_pred = kmeans_elbow1(*fool.features(), plot=True, repeat=10)
draw_parallel_coordinates(fool.feature, y_pred)
fool.feature["Cluster"]=y_pred
fool.feature_backup["Cluster"]=y_pred
### model selection end--------------------
feature_param_perCluster = fool.feature_backup.groupby('Cluster')["ave. phase B current(ss)"].mean()
# print(f"feature_param_perCluster:\n{feature_param_perCluster}")

# for j in range(0,2,1):
#     signal_composition(fool, event_param="curnt_B", feature_param=fool.param_feature_dict["curnt_B"][1], reduce=True, index=j)
