import unittest
import numpy as np
import pandas as pd
from pyEnergy.fool import Fool
from pyEnergy.clusters.two_stage import TwoStageCluster
import pyEnergy.CONST as CONST

class TestTwoStageCluster(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        self.data = pd.DataFrame({
            'real_power': np.random.rand(100),
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        self.csv_path = 'test_data.csv'
        self.data.to_csv(self.csv_path)
        self.fool = Fool(self.csv_path)
        self.fool.feature = self.data
        self.fool.feature_backup = self.data.copy()
        
    def test_init(self):
        """测试初始化功能"""
        # 测试默认参数
        model = TwoStageCluster(self.fool)
        self.assertEqual(model.method, 'hac')
        self.assertEqual(model.linkage, 'ward')
        
        # 测试不同聚类方法
        methods = ['hac', 'kmeans', 'gaussian']
        for method in methods:
            model = TwoStageCluster(self.fool, method=method)
            self.assertEqual(model.method, method)
        
        # 测试无效的聚类方法
        with self.assertRaises(ValueError):
            TwoStageCluster(self.fool, method='invalid')
    
    def test_first_stage(self):
        """测试第一阶段聚类"""
        model = TwoStageCluster(self.fool)
        params = {
            'min_samples': 2,
            'max_clusters': 5,
            'repeats': 2,
            'weights': [0.5, 0.5]
        }
        
        # 测试基本功能
        y_pred = model._perform_first_stage(**params)
        self.assertIsNotNone(y_pred)
        self.assertEqual(len(y_pred), len(self.data))
        self.assertIsNotNone(model.stage1_model)
    
    def test_second_stage(self):
        """测试第二阶段聚类"""
        model = TwoStageCluster(self.fool)
        params = {
            'min_samples': 2,
            'max_clusters': 5,
            'repeats': 2,
            'weights': [0.5, 0.5]
        }
        
        # 先执行第一阶段聚类
        model._perform_first_stage(**params)
        
        # 测试基本功能
        non_power_features = ['feature1', 'feature2']
        y_pred = model._perform_second_stage(non_power_features, **params)
        self.assertIsNotNone(y_pred)
        self.assertEqual(len(y_pred), len(self.data))
        
        # 测试空特征情况
        with self.assertRaises(ValueError):
            model._perform_second_stage([], **params)
    
    def test_fit(self):
        """测试完整的聚类过程"""
        model = TwoStageCluster(self.fool)
        params = {
            'min_samples': 2,
            'max_clusters': 5,
            'repeats': 2,
            'weights': [0.5, 0.5]
        }
        
        # 测试基本功能
        y_pred, score, n_clusters = model.fit(**params)
        self.assertIsNotNone(y_pred)
        self.assertEqual(len(y_pred), len(self.data))
        self.assertGreaterEqual(n_clusters, 1)
        
        # 测试聚类映射
        cluster_mapping = model.get_cluster_mapping()
        self.assertIsInstance(cluster_mapping, dict)
    
    def test_different_methods(self):
        """测试不同的聚类方法"""
        params = {
            'min_samples': 2,
            'max_clusters': 5,
            'repeats': 2,
            'weights': [0.5, 0.5]
        }
        
        methods = ['hac', 'kmeans', 'gaussian']
        for method in methods:
            model = TwoStageCluster(self.fool, method=method)
            y_pred, score, n_clusters = model.fit(**params)
            self.assertIsNotNone(y_pred)
            self.assertEqual(len(y_pred), len(self.data))
            self.assertGreaterEqual(n_clusters, 1)
    
    def test_edge_cases(self):
        """测试边界情况"""
        model = TwoStageCluster(self.fool)
        params = {
            'min_samples': 2,
            'max_clusters': 5,
            'repeats': 2,
            'weights': [0.5, 0.5]
        }
        
        # 测试单样本簇
        small_data = pd.DataFrame({
            'real_power': [1],
            'feature1': [1],
            'feature2': [1]
        })
        self.fool.feature = small_data
        self.fool.feature_backup = small_data.copy()
        y_pred, score, n_clusters = model.fit(**params)
        self.assertEqual(len(y_pred), 1)
        
        # 测试空特征数据
        empty_data = pd.DataFrame()
        self.fool.feature = empty_data
        self.fool.feature_backup = empty_data.copy()
        with self.assertRaises(Exception):
            model.fit(**params)
    
    def tearDown(self):
        """测试后的清理工作"""
        import os
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

if __name__ == '__main__':
    unittest.main()