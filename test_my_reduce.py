import pickle

import pandas as pd
from pyEnergy.drawer import draw_corr, draw_signal, draw_signal_series
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
###* ----------feature selection start------------------
# fool = initialize_with_feature_selector(
#     path, method="corr",
#     selection_params={"n_components": 5, "threshold": 0.3}, drought=True)

with open('fool.pkl', 'rb') as f:
    fool = pickle.load(f)
data = fool.other_event[12]
draw_signal(data, param_str="curnt_B", ax=None)
from pyEnergy.composition.reducer import GassianSmoothing
reducer = GassianSmoothing()
signal, signal2 = reducer(data['curnt_B'])
index = data.index  # 或者其他适当的索引
signal2_series = pd.Series(signal2, index=index)
draw_signal_series(signal2_series)


