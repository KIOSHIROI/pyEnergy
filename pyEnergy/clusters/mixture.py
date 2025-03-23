import numpy as np
from pyEnergy.clusters.model import Model, compute_score
from sklearn.mixture import GaussianMixture as gmm
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import KFold
import os
import warnings

# 设置环境变量以避免内存泄漏
os.environ['OMP_NUM_THREADS'] = '1'
# 过滤KMeans内存泄漏警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

class Gaussian(Model):
    def __init__(self, fool):
        super().__init__(fool)
        self.use(gmm)
        self.model_name = "Gaussian"
        self.best_model = None
        self.scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
    
    def _select_best_model(self, X, n_components_range=range(2, 11), 
                          covariance_types=['full', 'tied', 'diag', 'spherical']):
        best_score = -np.inf
        best_model = None
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for n_components in n_components_range:
            for covariance_type in covariance_types:
                try:
                    cv_scores = []
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        
                        model = gmm(n_components=n_components,
                                   covariance_type=covariance_type,
                                   random_state=42,
                                   n_init=10)
                        model.fit(X_train)
                        
                        # 计算综合评分
                        labels = model.predict(X_val)
                        if len(np.unique(labels)) > 1:
                            silhouette = silhouette_score(X_val, labels)
                            calinski = calinski_harabasz_score(X_val, labels)
                            bic = -model.bic(X_val)  # 取负值使其与其他指标方向一致
                            aic = -model.aic(X_val)  # 取负值使其与其他指标方向一致
                            
                            # 归一化各个指标
                            score = (silhouette + calinski/1000 + bic/10000 + aic/10000) / 4
                            cv_scores.append(score)
                    
                    if cv_scores and np.mean(cv_scores) > best_score:
                        best_score = np.mean(cv_scores)
                        best_model = gmm(n_components=n_components,
                                        covariance_type=covariance_type,
                                        random_state=42,
                                        n_init=10)
                        best_model.fit(X)
                except Exception as e:
                    continue
        
        return best_model
    
    def fit(self, **params):
        # 数据预处理：标准化和异常值处理
        X = self.standard_scaler.fit_transform(self.fool.feature)
        X = self.scaler.fit_transform(X)
        
        # 自动选择最佳模型参数
        self.best_model = self._select_best_model(X)
        
        if self.best_model is None:
            raise ValueError("无法找到合适的模型参数")
        
        # 获取聚类结果
        self.y_pred = self.best_model.predict(X)
        
        # 计算评估指标
        if len(np.unique(self.y_pred)) > 1:
            silhouette = silhouette_score(X, self.y_pred)
            calinski = calinski_harabasz_score(X, self.y_pred)
            bic = self.best_model.bic(X)
            aic = self.best_model.aic(X)
            
            print(f"聚类评估指标:")
            print(f"轮廓系数: {silhouette:.4f}")
            print(f"Calinski-Harabasz指数: {calinski:.4f}")
            print(f"BIC: {bic:.4f}")
            print(f"AIC: {aic:.4f}")
            
            self.score = silhouette
        else:
            self.score = -1
        
        print(f"最佳模型参数: n_components={self.best_model.n_components}, "
              f"covariance_type={self.best_model.covariance_type}")
        
        n_clusters = len(np.unique(self.y_pred))
        return self.y_pred, self.score, n_clusters
    
    def predict(self, X):
        if self.best_model is None:
            raise ValueError("模型尚未训练")
        X_scaled = self.standard_scaler.transform(X)
        X_scaled = self.scaler.transform(X_scaled)
        return self.best_model.predict(X_scaled)



