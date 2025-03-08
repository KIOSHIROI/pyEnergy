import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pyEnergy.CONST as CONST
from pyEnergy.fool import Fool
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from auto_test import auto_test

# 数据路径配置
path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
print(f"使用数据文件: {path}")

# 特征选择配置
OLD_FEATURES_1 = [CONST.feature_info[1], CONST.feature_info[9]]

# 创建原始fool对象
# fool = Fool(path).select("selected", selected_features=OLD_FEATURES_1)

# # 测试单独的聚类模型
# print("\n=== 测试HAC模型 ===")
# auto_test(fool, HAC)

# print("\n=== 测试高斯混合模型 ===")
# auto_test(fool, Gaussian)

fool = Fool(path).select("selected", selected_features=CONST.REAL_POWER_FEATURES)
# # 测试两阶段聚类（相同方法）
# print("\n=== 测试HAC两阶段聚类 ===")
# auto_test(fool, HAC, is_two_stage=True, weights=[0.5, 0.5])

# print("\n=== 测试高斯混合两阶段聚类 ===")
# auto_test(fool, Gaussian, is_two_stage=True, weights=[0.5, 0.5])

# 测试两阶段聚类（不同方法组合）
print("\n=== 测试HAC+高斯混合两阶段聚类 ===")
auto_test(fool, HAC, is_two_stage=True, weights=[0.25, 0.75], stage2_method=Gaussian)

# print("\n=== 测试高斯混合+HAC两阶段聚类 ===")
# auto_test(fool, Gaussian, is_two_stage=True, weights=[0.5, 0.5], stage2_method=HAC)