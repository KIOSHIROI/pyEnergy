import numpy as np
import glob, os
import matplotlib.pyplot as plt
import old_files.utils_EDDA as utils_EDDA
import myutils

###############################################################################
# Import data 
folder = 'data'
data_files = os.path.join(folder + "/WuFanZhuang-45-139040-values 20181001-20190214.csv")

df = myutils.import_transformer_data(data_files)

## preprocess the data;
## opt 1: set <1 kW to 0
#minimum_active_power = 1 # [kW]
#Isless_than_one = df['RealP'] < minimum_active_power
#df['RealP'][Isless_than_one] = 0
#df.loc[:,'step'] = df.index.to_series().diff().astype('timedelta64[m]')


###############################################################################
# Get all events. One event is an incident of single/multi pumping operation.
event_duration, events_all = myutils.find_all_events(df, 1, 10)

###############################################################################
# Get single pump events 
monotypeEvents, idx_monotype, otherEvents = utils_EDDA.find_monotypeEvents(events_all, 1)

ctr = 0
for event_ in monotypeEvents:
    
    event_ = utils_EDDA.estimate_tot_power(event_)
    monotypeEvents[ctr] = event_
    
    ctr += 1
#
# ###############################################################################
# # Compute feature lists
# feature_list, idx_used_events = utils_EDDA.compute_features(monotypeEvents)
#
# ###############################################################################
# # Apply clustering algorithm
#
# # list of features:
# #
#     # 'std. real power(ss)'
#     # 'ave. real power(ss)'
#     # 'max. real power(tr)'
#     # 'std. reactive power(ss)'
#     # 'ave. reactive power(ss)'
#     # 'max. reactive power(tr)'
#     # 'std. phase B current(ss)'
#     # 'ave. phase B current(ss)'
#     # 'max. phase B current(tr)'
#
# #TODO: debug the normalize option
# feature_selected = ['ave. real power(ss)', 'ave. reactive power(ss)']
# grps, scores = utils_EDDA.KMeans_elbow(feature_list[feature_selected].values,
#                                        normalize = False
#                                        )
#
# ###############################################################################
# # Apply Matching Pursuit (MP)
#
# thr_val = 5
#
# event = otherEvents[8]
#
# param_str = 'curnt_B'
# signals = event[param_str].values
#
# idx_jump = np.where( np.abs( np.diff( signals) ) > thr_val )[0]
# idx_jump = np.hstack( (0, idx_jump, len(signals)) )
#
# nr_pluses = len(idx_jump) - 1
#
# signals_compressed = np.zeros(nr_pluses)
#
# for i in range(0, nr_pluses):
#     idx1 = idx_jump[i]+1  # skip one element to exclude transition states
#     idx2 = idx_jump[i+1]
#
#     signals_compressed[i] = np.mean( signals[idx1:idx2] )
#
# nr_sources  = len( np.unique(grps) )
# nr_features = len( feature_selected )
#
# n_nonzero_coefs = 17

#TODO: for each signals, apply MP algorithm
# omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
# omp.fit(X, y)
# coef = omp.coef_


from sklearn.linear_model import OrthogonalMatchingPursuit
#from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
#
#nr_sources  = len(np.unique(grps)) 
#nr_features = 2
#
# n_components, n_features = 5, 2
# n_nonzero_coefs = 1
# #
# y, X, w = make_sparse_coded_signal(n_samples=1,
#                                    n_components=n_components,
#                                    n_features=n_features,
#                                    n_nonzero_coefs=n_nonzero_coefs,
#                                    random_state=0)
#
# idx, = w.nonzero()
#
# # plot the sparse signal
# plt.figure(figsize=(7, 7))
# plt.subplot(4, 1, 1)
# plt.xlim(0, n_components)
# plt.title("Sparse signal")
# plt.stem(idx, w[idx])
# #
# omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
# omp.fit(X, y)
# coef = omp.coef_
# idx_r, = coef.nonzero()
# plt.subplot(4, 1, 2)
# plt.xlim(0, n_components)
# plt.title("Recovered signal from noise-free measurements")
# plt.stem(idx_r, coef[idx_r])
#

###############################################################################
# Test
#
# plt.scatter(feature_list['ave. real power(ss)'],feature_list[ 'ave. reactive power(ss)'], c=grps)
