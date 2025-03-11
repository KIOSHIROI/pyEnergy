import numpy as np
from scipy.signal import find_peaks

def find_all_events(df, thre_val=1, thre_time=10,  param_str = 'curnt_B'):
    """
    从时间序列数据中查找并提取所有有效事件。
    
    参数:
        df: DataFrame, 包含电力数据的时间序列
        thre_val: float, 信号阈值，低于此值的信号被视为无效
        thre_time: float, 事件持续时间阈值（分钟），超过此值的事件被视为有效
        param_str: str, 用于事件检测的参数列名，默认为'curnt_B'
    
    返回:
        events_all: list, 所有有效事件的列表
        valid_event_durations: array, 有效事件的持续时间
    """
    # 复制信号数据并将低于阈值的信号置零
    signals = df[param_str].copy()
    idx_invalid = signals < thre_val
    signals[idx_invalid] = 0

    # 识别事件的起始和结束位置
    idx_valid = (signals > 0).astype(int)
    idx_valid = np.insert(idx_valid, 0, 0)  # 在开头插入0以便计算差分
    idx_diff = np.diff(idx_valid)  # 计算差分以找到状态变化点
    idx_start_event = np.where(idx_diff == 1)[0]  # 找到事件开始点
    idx_end_event = np.where(idx_diff == -1)[0] - 1  # 找到事件结束点

    # 确保每个开始事件都有对应的结束事件
    if len(idx_start_event) > len(idx_end_event):
        idx_start_event = idx_start_event[0:len(idx_end_event)]  # 截取相等数量的开始和结束事件

    # 计算每个事件的持续时间（分钟）
    event_durations = (df.index[idx_end_event] - df.index[idx_start_event]).total_seconds() / 60

    # 根据持续时间筛选有效事件
    valid_events_mask = event_durations > thre_time
    valid_event_durations = event_durations[valid_events_mask]
    valid_start_idx = idx_start_event[valid_events_mask]
    valid_end_idx = idx_end_event[valid_events_mask]

    # 提取有效事件数据
    events_all = [df.iloc[start:end+1] for start, end in zip(valid_start_idx, valid_end_idx)]

    # 过滤功率因数全为1的事件（这些可能是无效或异常事件）
    filtered_events = []
    filtered_durations = []
    for event, duration in zip(events_all, valid_event_durations):
        factor_B = event['factor_B'].values
        factor_C = event['factor_C'].values
        if not (np.all(factor_B == 1) or np.all(factor_C == 1)):
            filtered_events.append(event)
            filtered_durations.append(duration)
    events_all = filtered_events
    valid_event_durations = np.array(filtered_durations)

    return events_all, valid_event_durations


def identify_pulse(signal, min_peak_height=5):
    """
    识别信号中的脉冲数量。
    
    参数:
        signal: array, 输入信号
        min_peak_height: float, 最小峰值高度阈值
    
    返回:
        int: 检测到的脉冲数量
    """
    peaks, _ = find_peaks(signal, height=min_peak_height)
    return len(peaks)

def extract_monotype_events(events_all, thre_val=1, param_str='curnt_B'):
    """
    将所有有效事件分为"单一类型事件"和其他事件。
    单一类型事件是指信号变化较小或只有单个脉冲的事件。
    
    参数:
        events_all: list, 所有事件的列表
        thre_val: float, 信号方差阈值
        param_str: str, 用于分析的参数列名
    
    返回:
        monotype_events: list, 单一类型事件列表
        idx_monotype: array, 单一类型事件的布尔索引
        other_events: list, 其他类型事件列表
    """
    monotype_events = []
    other_events = []
    idx_monotype = np.zeros(len(events_all), dtype=bool)

    # 遍历所有事件，根据信号方差和脉冲数量进行分类
    for i, event_ in enumerate(events_all):
        signal = event_[param_str].values
        var_signal = np.var(signal)  # 计算信号方差
        nr_pulse = identify_pulse(signal, 5)  # 识别脉冲数量
        if var_signal < thre_val or nr_pulse == 1:  # 判断是否为单一类型事件
            monotype_events.append(event_)
            idx_monotype[i] = True
        else:
            other_events.append(event_)
    return monotype_events, idx_monotype, other_events


def estimate_total_power(event):
    """
    估计设备的总功率，包括有功功率和无功功率。
    基于三相电压、电流和功率因数计算。
    
    参数:
        event: DataFrame, 包含电压、电流和功率因数数据的事件
    
    返回:
        event: DataFrame, 添加了估算功率的事件数据
    """
    # 提取三相电压数据 [V]
    v_A = event['volt_A'].values
    v_B = event['volt_B'].values
    v_C = event['volt_C'].values

    # 提取相电流数据 [A]
    c_B = event['curnt_B'].values
    c_C = event['curnt_C'].values

    # 提取功率因数数据 [-]
    pf_A = event['factor_A'].values
    pf_B = event['factor_B'].values
    pf_C = event['factor_C'].values

    # 计算平均值
    ave_pf = np.mean([pf_A, pf_B, pf_C], axis=0)  # 平均功率因数
    ave_curnt = np.mean([c_B, c_C], axis=0)  # 平均电流
    ave_volt = np.mean([v_A, v_B, v_C], axis=0)  # 平均电压

    # 计算有功功率和无功功率 [kW, kVar]
    realP_proxy = 3 * ave_pf * ave_curnt * ave_volt / 1000  # 有功功率
    reactP_proxy = 3 * np.sqrt(1 - np.square(ave_pf)) * ave_curnt * ave_volt / 1000  # 无功功率

    # 将计算结果添加到事件数据中
    event = event.assign(realP_proxy=realP_proxy, reactP_proxy=reactP_proxy)

    return event

