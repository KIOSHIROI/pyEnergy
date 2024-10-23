from pyEnergy.cluster import kmeans_elbow
from pyEnergy.fool import initialize
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value 

def reduce_signals(signals, threshold):
    """对信号进行平滑处理，这里使用 Savitzky-Golay 滤波器来代替 MATLAB 的平滑"""
    smoothed_signals = savgol_filter(signals, window_length=7, polyorder=3)
    return np.where(smoothed_signals > threshold, smoothed_signals, 0), smoothed_signals

def signal_composition_opt1(realP_perClster, signal_reduced):
    """使用 MILP 优化将信号分解为不同聚类信号的线性组合"""
    sols = []
    n_clusters = realP_perClster.shape[0]
    
    # 遍历每个时刻的信号进行优化
    for signal in signal_reduced:
        # 定义 MILP 问题
        prob = LpProblem("Signal Decomposition", LpMinimize)
        
        # 定义优化变量（线性组合的系数）
        x = [LpVariable(f'x_{i}', lowBound=0) for i in range(n_clusters)]
        
        # 目标函数：最小化 (targetP - CX)
        prob += lpSum([(realP_perClster[i] * x[i]) for i in range(n_clusters)]) - signal
        
        # 求解问题
        prob.solve()
        
        # 保存最优解
        sols.append([value(x[i]) for i in range(n_clusters)])
    
    return sols

# 初始化
fool = initialize("data/ChangErZhai-40-139079-values 20180101-20181031.csv", 
                 feature_selection_method='pca',
                 selection_params={"n_components": 4})
y_pred = kmeans_elbow(*fool.features(), plot=False)

fool.feature["Cluster"] = y_pred
feature = fool.feature
realP_perClster = feature.groupby('Cluster')[10].mean()
workingP_perClster = feature.groupby('Cluster')[2].mean()

other_events = fool.other_event
residues_final = np.zeros(len(other_events))  # 用于存储每个事件的残差
thr_val = 5

# 遍历每个事件
for ii, eventSelected in enumerate(other_events):
    signals = eventSelected['curnt_B']  # 提取当前事件的信号
    
    # 平滑信号
    signal_reduced, _ = reduce_signals(signals, thr_val)
    
    # 使用 MILP 进行信号分解
    sols = signal_composition_opt1(realP_perClster, signal_reduced)
    
    # 计算每个脉冲的残差
    residue_perPulse_norm = np.zeros(len(sols))
    nr_types = realP_perClster.shape[0]  # 聚类的类别数

    for j in range(len(residue_perPulse_norm)):
        reconstructed_signal = np.dot(sols[j], realP_perClster)
        residue_perPulse_norm[j] = abs((reconstructed_signal - signal_reduced[j]) / signal_reduced[j])

    # 计算该事件的平均残差
    residues_final[ii] = np.mean(residue_perPulse_norm)

# 绘制残差
plt.figure()
plt.plot(residues_final * 100, linewidth=1.5, color=[0/255, 108/255, 155/255])
plt.grid(True)
plt.xlim([0, len(residues_final) + 1])
plt.ylim([0, 100])
plt.ylabel('Absolute error percentage [%]')
plt.xlabel('Event ID')
plt.show()
