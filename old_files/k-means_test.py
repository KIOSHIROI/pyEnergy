from pyEnergy.core import *
from pyEnergy.check import *
from pyEnergy.compute import *
from pyEnergy.drawer import *
import pandas as pd
import matplotlib.pyplot as plt


df = import_transformer_data("data/ChangErZhai-40-139079-values 20180101-20181031.csv")

event_all, _ = find_all_events(df)

monotype_event, idx_monotype_event, other_event = extract_monotype_events(event_all)

feature, _ = compute_features(monotype_event)

feature_standardized = standard(feature)
feature_normalized = normal(feature)
# feature_standardized.to_csv("data/feature_standardized_ChangErZhai.csv")
feature_info = [
    'std. real power(ss)', 'ave. real power(ss)', 'max. real power(tr)',
    'std. reactive power(ss)', 'ave. reactive power(ss)', 'max. reactive power(tr)',
    'std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)'
]


