import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA as p

# 扩展的特征选择函数
def selection(selector):
    selectors = {
        'selected': Selected,
        'pca': PCA,
        'corr': Correlation,
        'remove': Remover,
        'kbest': KBestSelector,
        'l1': L1SparseSelector,
        'variance': VarianceSelector,
        'autoencoder': AutoencoderSelector  # 添加自编码器选择器
    }
    if selectors.get(selector):
        return selectors[selector]
    raise ValueError("Invalid feature selection method or missing parameters.")

# 基础选择器类
class Selector:
    def __init__(self):
        pass
    
    def __call__(self, features, **params):
        return self.select(features, **params)
    
    def select(self, features, **params):
        pass

# 手动选择特征
class Selected(Selector):
    def __init__(self):
        super().__init__()
    def select(self, features, **params):
        selected_features = params.get("selected_features", [])

        if not selected_features:
            print("No features selected.")
            return features, []
        selected_features = features[selected_features]
        return selected_features, selected_features.columns


# 删除指定特征的选择器
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

# PCA特征选择器
class PCA(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        n_components = params.get("n_components", 5)
        scaler = StandardScaler()
        pca = p(n_components=n_components)
        
        pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
        pca_list = []
        transformed_features = pipeline.fit_transform(features)
        print("Selected features (PCA-based):")
        for i in range(n_components):
            print(f"Principal Component {i + 1}")
            pca_list.append(f"PC{i}")

        return transformed_features, pca_list

# 相关性特征选择器
class Correlation(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        threshold = params.get("threshold", 0.5)
        print("Threshold={}".format(threshold))
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        selected_features = features.drop(columns=to_drop)

        print("Selected features (correlation-based):")
        for feature in selected_features.columns:
            print(feature)
        
        return selected_features, selected_features.columns

# KBest特征选择器
class KBestSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        k = params.get("k", 5)
        score_func = params.get("score_func", f_classif)  # 默认使用方差分析F值作为评分函数
        selector = SelectKBest(score_func, k=k)
        selected_features = selector.fit_transform(features, features)  # 无监督任务，目标与特征相同
        selected_columns = features.columns[selector.get_support()]
        print("Selected features (KBest-based):", selected_columns)

        return selected_features, selected_columns

# L1正则化（Lasso）特征选择器
class L1SparseSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        alpha = params.get("alpha", 0.1)  # Lasso的正则化强度
        lasso = Lasso(alpha=alpha)
        lasso.fit(features, features)  # 无监督任务，目标与特征相同
        selected_features = features.columns[lasso.coef_ != 0]
        print("Selected features (L1 regularization):", selected_features)

        return features[selected_features], selected_features

# 方差特征选择器
class VarianceSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, features, **params):
        threshold = params.get("threshold", 0.01)  # 设置方差阈值，低于该阈值的特征将被移除
        selector = VarianceThreshold(threshold=threshold)
        selected_features = selector.fit_transform(features)
        selected_columns = features.columns[selector.get_support()]
        print("Selected features (Variance-based):", selected_columns)

        return selected_features, selected_columns

class AutoencoderSelector(Selector):
    def __init__(self):
        super().__init__()

    # 构建自编码器网络
    def build_autoencoder(self, input_dim, encoding_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # 解码层使用Sigmoid
        )
        return model

    def select(self, features, **params):
        encoding_dim = params.get("encoding_dim", 5)  # 压缩后的维度（潜在空间）
        epochs = params.get("epochs", 50)  # 训练轮次
        batch_size = params.get("batch_size", 32)  # 每批数据大小
        learning_rate = params.get("learning_rate", 0.001)

        # 确保输入数据是DataFrame类型并转换为numpy数组
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        
        # 标准化数据
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 转换为PyTorch Tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 确定使用设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建自编码器
        model = self.build_autoencoder(input_dim=features.shape[1], encoding_dim=encoding_dim).to(device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练自编码器
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            reconstructed = model(features_tensor.to(device))  # 移动到GPU
            loss = criterion(reconstructed, features_tensor.to(device))
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # 获取编码器的输出（潜在空间表示）
        with torch.no_grad():
            encoder = nn.Sequential(*list(model.children())[:2])  # 只保留编码部分
            encoded_features = encoder(features_tensor.to(device)).cpu().numpy()

        # 打印选择的特征（压缩后的潜在空间）
        selected_columns = [f"PC{i+1}" for i in range(encoding_dim)]
        print("Selected features (Autoencoder-based):", selected_columns)

        return encoded_features, selected_columns