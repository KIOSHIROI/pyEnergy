from pyEnergy.cluster import kmeans, kmeans_elbow1
from pyEnergy.final import signal_composition
from pyEnergy.fool import initialize_with_feature_selector
from pyEnergy.drawer import draw_corr

path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
fool = initialize_with_feature_selector(path, method="corr", selection_params={"n_components": 8, "threshold": 0.7})
# fool.box()
#? fool.feature_info = [fool.feature.columns[1], fool.feature.columns[9]]
#? fool.feature = fool.feature[[fool.feature.columns[1], fool.feature.columns[9]]] # 使用1和9（对应matlab 2,10）
#? draw_corr(feature=fool.feature) # 绘制相关图 发现相关系数高达0.98
###* ----------feature selection end--------------------
###* ----------model selection start--------------------

y_pred = kmeans(*fool.features(), plot=False, repeat=100, weight=[0.0, 1])
print(y_pred)
# fool.feature_info = [fool.feature.columns[0],fool.feature.columns[1],fool.feature.columns[2], fool.feature.columns[3]]
# fool.feature = fool.feature[fool.feature_info]
# draw_corr(feature=fool.feature)
# y_pred = kmeans(*fool.features(), plot=True, repeat=50, weight=[0.5, 0.5])
# fool.feature["Cluster"]=y_pred
# fool.feature_backup["Cluster"]=y_pred
# import pickle
# with open('kmeansPenal.pkl', 'wb') as f:
#     pickle.dump(fool, f)
# ###* ----------model selection end----------------------

# max = len(fool.other_event)
# for j in range(1, 2):
#     print("MEAN")
#     #! reduce = "gaussian" | "moving" | "wavelet" | ""
#     reduce = "wavelet"
#     sols, err = signal_composition(fool, event_param="realP_B", feature_param=fool.param_feature_dict["realP"][1], 
#                                    reduce="gaussian", reduce_params={"sigma": 0.1}, index=j, up_bound=None, n_color=7, color_reverse=True,
#                                 #    plot_original=False, 
#                                 #    plot=False, 
#                                 #    save=f"images/reduces/changer40/{j}_{reduce}_.png"
#                                    )
    # print("MEDIAN")
    # sols, err = signal_composition(fool, event_param="curnt_B", feature_param=fool.param_feature_dict["curnt_B"][1], reduce=True, index=j, up_bound=None, group_method="median")
    # print("err:", err)

