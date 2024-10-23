import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

def filter_based_selection(features, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    selected_features = selector.fit_transform(features)
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = selected_features[selected_indices]

    print("Selected features (filter-based):")
    for feature in selected_feature_names:
        print(feature)

    return features.iloc[:, selected_indices], selected_indices


def pca_based_selection(features, n_components=5):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca)
    ])
    
    transformed_features = pipeline.fit_transform(features)
    print("Selected features (PCA-based):")
    for i in range(n_components):
        print(f"Principal Component {i + 1}")

    return transformed_features, pca


def correlation_based_selection(features, threshold=0.9):
    correlation_matrix = features.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    selected_features = features.drop(columns=to_drop)
    selected_indices = [i for i in range(features.shape[1]) if features.columns[i] not in to_drop]

    print("Selected features (correlation-based):")
    for feature in selected_features.columns:
        print(feature)

    return selected_features, selected_indices
