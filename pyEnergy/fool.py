from matplotlib import pyplot as plt
from pyEnergy.check import import_transformer_data
from pyEnergy.compute import min_max as z_score
from pyEnergy.preprocess.feature_computer import compute_features
from pyEnergy.core import extract_monotype_events, find_all_events
from pyEnergy.preprocess.feature_selector import selection
import pandas as pd

class Fool:
    def __init__(self, csv, normal=True, drought=False):
        df = import_transformer_data(csv)
        self.csv = csv
        event_all, _ = find_all_events(df)
        self.monotype_event, idx_monotype_event, self.other_event = extract_monotype_events(event_all)
        feature, _ = compute_features(self.monotype_event, drought)
        self.feature = feature
        self.feature_backup = feature.copy()
        self.original_data = df
        
        if normal:
            self.feature = z_score(feature)
        
        if 'Total W' in self.feature.columns:
            self.power_feature = self.feature['Total W'].values.reshape(-1, 1)
        else:
            self.power_feature = self.feature.iloc[:, 0].values.reshape(-1, 1)
        self.other_features = self.feature.drop(columns=[self.feature.columns[0]]) 
        
        print("fool init.")

    def select(self, method, **params):
        selector = selection(method)()
        selected_features, feature_info = selector(self.feature, **params)
        feature = pd.DataFrame(selected_features, columns=feature_info)
        self.feature = feature
        self.feature_info = feature_info
        print("-" * 10)
        return self

    def features(self):
        return self.feature, self.feature_info

    def box(self):
        self.feature.plot.box()
        plt.show()