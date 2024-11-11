from matplotlib import pyplot as plt
from pyEnergy.check import import_transformer_data
# from pyEnergy.compute import z_score
from pyEnergy.compute import min_max as z_score
from pyEnergy.preprocess.feature_computer import compute_features
from pyEnergy.core import extract_monotype_events, find_all_events
from pyEnergy.preprocess.feature_selector import selection
import pandas as pd

class Fool:       
    def __init__(self, csv, normal=True, drought=False):
        df = import_transformer_data(csv)
        event_all, _ = find_all_events(df)
        self.monotype_event, idx_monotype_event, self.other_event = extract_monotype_events(event_all)
        feature, _ = compute_features(self.monotype_event, drought)
        self.feature_backup = feature
        if normal:
            self.feature = z_score(feature)

    def select(self, method, **params):
        selector = selection(method)()
        selected_features, feature_info = selector(self.feature, **params)
        feature = pd.DataFrame(selected_features, columns=feature_info)
        self.feature = feature
        self.feature_info = feature_info
        return self
    def features(self):
        return self.feature, self.feature_info 
        
    def box(self):
        self.feature.plot.box()
        plt.show()
