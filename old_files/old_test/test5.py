import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value

# # 示例数据
# realP_perCluster = np.array([10, 20, 30])  # 每个聚类的平均值
# signal_reduced = np.array([60, 45, 30])     # 要分解的信号

def signal_composition_opt1(realP_perCluster, signal_reduced):
    sols = []
    n_clusters = len(realP_perCluster)  # 聚类数量
    
    for signal in signal_reduced:
        prob = LpProblem("Signal_Decomposition", LpMinimize)
        
        # 定义优化变量
        x = [LpVariable(f'x_{i}', lowBound=0, cat='Integer') for i in range(n_clusters)]
        error = LpVariable('error', lowBound=0)  # 辅助变量表示误差

        # 目标函数：最小化误差
        prob += error + lpSum(x)

        # 添加约束：线性组合的值必须等于信号加上误差
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) + error >= signal
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) - error <= signal

        # 求解问题
        prob.solve()
        
        # 保存最优解
        sols.append([value(x[i]) for i in range(n_clusters)])
    
    return sols

# # 执行函数
# solutions = signal_composition_opt1(realP_perCluster, signal_reduced)

# # 输出解决方案
# print("最优解:")
# for i, sol in enumerate(solutions):
#     print(f"时间点 {i}: {sol}")
