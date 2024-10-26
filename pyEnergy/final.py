
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
from pyEnergy import drawer
from scipy.ndimage import gaussian_filter
import pywt

def my_reduce(signal, threshold=1, delta=1):
    step = 3
    window = 5
    for i in range(0, len(signal), 1):
        if len(signal) - i < 3: 
            break
        if np.abs(signal[i+2] - signal[i]) < threshold and np.abs(signal[i+2] + signal[i] - 2*signal[i+1]) > delta:
            signal[i+1] = signal[i] + (signal[i+2] - signal[i]) / 2

    for i in range(0, len(signal), 1):
        if len(signal) - i < 4: 
            break
        if np.abs(signal[i+3] - signal[i]) < threshold and np.abs(signal[i+3] + signal[i] - (signal[i+1]+signal[i+2])/2) > delta:
            signal[i+1] = signal[i] + (signal[i+3] - signal[i]) / 2
            signal[i+2] = signal[i] + (signal[i+3] - signal[i]) / 2
    return signal



def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def gaussian_smoothing(signal, sigma=1):
    return gaussian_filter(signal, sigma=sigma)

def wavelet_smoothing(signal, wavelet='db4', level=None, threshold=None):
    coeff = pywt.wavedec(signal, wavelet, level=level)
    if threshold == None:
        threshold = np.median(np.abs(coeff[-1])) / 0.6745
    coeff[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeff[1:]]  # 去噪

    return pywt.waverec(coeff, wavelet)

def reduce_signal1(signals, threshold=5):
    """对信号进行平滑处理，这里使用 Savitzky-Golay 滤波器来代替 MATLAB 的平滑"""
    smoothed_signals = savgol_filter(signals, window_length=5, polyorder=3)
    return np.where(smoothed_signals > threshold, smoothed_signals, 0), smoothed_signals

def reconstruct_time_series(sols, phaseB_perCluster):
    reconstructed = []
    for sol in sols:
        reconstructed_signal = sum(phaseB_perCluster.iloc[i] * sol[i] for i in range(len(sol)))
        reconstructed.append(reconstructed_signal)
    return np.array(reconstructed)

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD

def signal_composition_opt1(realP_perCluster, signal_reduced, low_bound=0, up_bound=1):
    sols = []
    errors = []  # 用于保存每个信号的误差
    n_clusters = len(realP_perCluster)  # 聚类数量
    
    for signal in signal_reduced:
        prob = LpProblem("Signal_Decomposition", LpMinimize)
        
        # 定义优化变量
        x = [LpVariable(f'x_{i}', lowBound=low_bound, upBound=up_bound, cat='Integer') for i in range(n_clusters)]
        error = LpVariable('error', lowBound=0)

        # 目标函数：最小化误差
        prob += error + lpSum(x)

        # 添加约束：线性组合的值必须等于信号加上误差
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) + error >= signal
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) - error <= signal

        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # 保存最优解和误差
        sols.append([int(value(x[i])) for i in range(n_clusters)])
        errors.append(value(error))  # 保存误差值
    
    return sols, errors  # 返回最优解和误差值


#TODO: signal reduce opt, draw division plot
def signal_composition(fool, event_param, feature_param, reduce=None, reduce_params={}, 
                       low_bound=0, up_bound=None, 
                       cmap="summer", n_color=10, color_reverse=False,
                       plot=True, plot_original=True, save=None, **kargs):
    other_events = fool.other_event
    if kargs.get("group_method")=="median":
        feature_param_perCluster = fool.feature_backup.groupby('Cluster')[feature_param].median()
    else:
        feature_param_perCluster = fool.feature_backup.groupby('Cluster')[feature_param].mean()
    if event_param == "realP_B":
        feature_param_perCluster /= 3
    print(f"{feature_param_perCluster}")
    n_clusters = feature_param_perCluster.shape[0]
    # print(n_clusters)
    idx = kargs.get("index", 0)
    event = other_events[idx]
    signal = event[event_param]
    x_values = signal.index
    if reduce == "" or None:
        signal1, signal2 = reduce_signal1(signal, 1)
        reduced1 = event_param + "reduced1"
        reduced2 = event_param + "reduced2"
        event[reduced1] = signal1
        event[reduced2] = signal2
        signal = signal1
        if plot_original == True:
            drawer.draw_signal(event, reduced1)
    else:
        if reduce == "gaussian":
            signal = gaussian_filter(signal, reduce_params.get("sigma", 1))
        elif reduce == "moving":
            signal = moving_average(signal, reduce_params.get("size", 5))
        elif reduce == "wavelet":
            wavelet = reduce_params.get("wavelet", "db4")
            threshold = reduce_params.get("threshold", 0.5)
            level = reduce_params.get("level", None)

            signal = wavelet_smoothing(signal, wavelet, level, threshold)
        elif reduce == "my":
            delta = reduce_params.get("delta", 1)
            threshold = reduce_params.get("threshold", 1)
            signal = my_reduce(signal, threshold, delta)
        if plot_original == True:
            drawer.draw_signal(event, event_param)  

    sols, errs = signal_composition_opt1(feature_param_perCluster, signal, low_bound=low_bound, up_bound=up_bound)
    print(sols)
    reconstructed_signal = reconstruct_time_series(sols, feature_param_perCluster)
    
    drawer.draw_continue_line_and_running_time(signal, reconstructed_signal, n_clusters, sols,
                                               x_values=x_values, 
                                               n_color=n_color, color_reverse=color_reverse, 
                                               plot=plot, save=save)
    return sols, errs