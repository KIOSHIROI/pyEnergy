import numpy as np
from sklearn.decomposition import PCA as p
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def selection(selector):
    selectors = {
        'pca': PCA,
        'corr': Correlation,
        'remove': Remover
    }
    for k, v in selectors.items():
        if selector == k:
            return v
    raise ValueError("Invalid feature selection method or missing parameters.")
    

class Selector:
    def __init__(self):
        pass
    
    def __call__(self, features, **params):
        return self.select(features, **params)
    
    def select(self, features, **params):
        pass

class Remover(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        remove_features = params.get("remove_features", [])
        
        if not remove_features:
            print("No features to remove.")
            return features, []

        selected_features = features.drop(columns=remove_features, errors='ignore')


        return selected_features, selected_features.columns

class PCA(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        n_components = params.get("n_components", 5)
        scaler = StandardScaler()
        pca = p(n_components=n_components)
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('pca', pca)
        ])
        pca_list = []
        transformed_features = pipeline.fit_transform(features)
        print("Selected features (PCA-based):")
        for i in range(n_components):
            print(f"Principal Component {i + 1}")
            pca_list.append(f"PC{i}")

        return transformed_features, pca_list

class Correlation(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        threshold = params.get("threshold", 0.5)
        print("threshold={}".format(threshold))
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        selected_features = features.drop(columns=to_drop)

        print("Selected features (correlation-based):")
        for feature in selected_features.columns:
            print(feature)
        
        return selected_features, selected_features.columns