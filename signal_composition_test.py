import pickle
from pyEnergy.final import signal_composition
from pyEnergy import drawer
with open('good_models/kmeans/kmeansPenal_corr4_thre0.6.pkl', 'rb') as f:
    fool = pickle.load(f)

max = len(fool.other_event)
for j in range(10, max):
    print("MEAN")
    #! reduce = "gaussian" | "moving" | "wavelet" | ""
    reduce="gaussian"
    sols, err = signal_composition(fool, event_param="realP_B", feature_param=fool.param_feature_dict["realP"][1], 
                                   reduce="gaussian", reduce_params={"sigma": 0.1}, index=j, up_bound=None, n_color=7, color_reverse=True,
                                   plot_original=False, 
                                   plot=False, 
                                   save=f"images/reduces/changer40/kmeansPenal_corr4_thre06/{j}_{reduce}_.png"
                                   )