from matplotlib import pyplot as plt
from pyEnergy.check import import_transformer_data
from pyEnergy.compute import standard, normalize
from pyEnergy.core import compute_features, extract_monotype_events, find_all_events
from pyEnergy.feature_selector import filter_based_selection, pca_based_selection, correlation_based_selection
import pandas as pd

feature_info = [
    'std. real power(ss)', 
    'ave. real power(ss)', 
    'trend real power(ss)', 
    'max. real power(tr)',

    'std. reactive power(ss)', 
    'ave. reactive power(ss)', 
    'trend reactive power(ss)', 
    'max. reactive power(tr)',
    
    'std. phase B current(ss)', 
    'ave. phase B current(ss)', 
    'trend phase B current(ss)', 
    'max. phase B current(tr)'
]

param_feature_dict = {
    "curnt_B": ['std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)'],
    "realP": ['std. real power(ss)', 'ave. real power(ss)', 'max. real power(tr)'],
    "reactP": ['std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)']
}

#Class Fool
class Fool:
    def __init__(self) -> None:
        self.feature = None
        self.feature_backup = None
        self.feature_info = None
        self.param_feature_dict = None
        self.monotype_event = None
        self.other_event = None
        
    def features(self):
        return self.feature, self.feature_info 
    
    def box(self):
        self.feature.plot.box()
        plt.show()

    def feature_selection(self, remove_feature=[], method=None, selection_params={}, normal=True):
        """
        特征选择函数，根据指定的方法选择特征。
        - method (str): 特征选择方法，可选 'filter', 'pca', 'correlation' 或 None。
        - selection_params (dict): 特征选择的参数，例如 {'threshold': 0.1} 用于过滤法。
        """
        if normal == True:
            self.feature = normalize(self.feature)

        feature = self.feature
        feature_info = self.feature_info
        
        for feature_ in remove_feature:
            feature_info.remove(feature_)
            feature = feature[feature_info]

        selected_features, feature_info = feature_selection_core(feature, feature_info, method, selection_params)
        feature = pd.DataFrame(selected_features, columns=feature_info)

        # 更新 Fool 对象的特征和特征信息
        self.feature = feature
        self.feature_info = feature_info



#Other Func
def initialize_with_feature_selector(csv, remove_feature=[], select_feature=None, method=None, selection_params={}, normal=True):
    """
    初始化函数，加载数据并进行特征提取和选择。
    - method (str): 特征选择方法，可选 'filter', 'pca', 'correlation' 或 None。
    - selection_params (dict): 特征选择的参数，例如 {'threshold': 0.1} 用于过滤法。
    """
    global feature_info
    # 导入并预处理数据
    df = import_transformer_data(csv)
    event_all, _ = find_all_events(df)

    fool = Fool()

    monotype_event, idx_monotype_event, other_event = extract_monotype_events(event_all)
    feature, _ = compute_features(monotype_event)
    fool.feature_backup = feature.copy()
    if normal == True:
        feature = standard(feature)
    # 移除指定特征
    for feature_ in remove_feature:
        feature_info.remove(feature_)
    # 指定选中特征
    if select_feature:
        feature_info = select_feature

    feature = feature[feature_info]

    # 特征选择
    print(feature_info)
    selected_features, feature_info = feature_selection_core(feature, feature_info, method, selection_params)
    feature = pd.DataFrame(selected_features, columns=feature_info)

    # 创建 Fool 对象并保存提取的特征和事件
    fool.feature = feature
    fool.monotype_event = monotype_event
    fool.feature_info = feature_info
    fool.other_event = other_event
    fool.param_feature_dict = param_feature_dict
    return fool

def initialize(csv, normal=True):
    """
    初始化函数，加载数据并进行特征提取。
    """
    global feature_info
    fool = Fool()
    # 导入并预处理数据
    df = import_transformer_data(csv)
    event_all, _ = find_all_events(df)

    monotype_event, idx_monotype_event, other_event = extract_monotype_events(event_all)
    feature, _ = compute_features(monotype_event)
    fool.feature_backup = feature
    if normal:
        feature = standard(feature)

    # 创建 Fool 对象并保存提取的特征和事件
    
    fool.feature = feature
    fool.monotype_event = monotype_event
    fool.feature_info = feature_info
    fool.other_event = other_event
    fool.param_feature_dict = param_feature_dict
    return fool


def feature_selection_core(feature, feature_info, method=None, selection_params={}):
    selected_features = feature
    if method is not None:
        if method == "filter":
            selected_features, selected_indices = filter_based_selection(feature, selection_params["threshold"]) #threshold
        elif method == "pca":
            n_components = selection_params.get('n_components', 5)
            selected_features, _ = pca_based_selection(feature, n_components=n_components)
            feature_info = [f'PC{i + 1}' for i in range(n_components)]  # 主成分不保留原特征名
        elif method == "corr":
            threshold = selection_params.get('threshold', 0.9)
            selected_features, selected_indices = correlation_based_selection(feature, threshold=threshold)
            feature_info = [feature_info[i] for i in selected_indices]
        elif method == "":
            pass
        else:
            raise ValueError("Invalid feature selection method or missing parameters.")
        
    return  selected_features, feature_info