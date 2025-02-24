import numpy as np
def find_all_events(df, thre_val=1, thre_time=10,  param_str = 'curnt_B'):

    signals = df[param_str].copy()
    idx_invalid = signals < thre_val
    signals[idx_invalid] = 0

    idx_valid = signals > 0
    idx_valid = np.insert(idx_valid.values.astype(int), 0, 0)
    idx_diff = np.diff(idx_valid)
    idx_start_event = np.where(idx_diff == 1)[0]
    idx_end_event = np.where(idx_diff == -1)[0] - 1

    # 确保每个开始事件都有对应的结束事件
    if len(idx_start_event) > len(idx_end_event):
        idx_start_event = idx_start_event[0:len(idx_end_event)]  # 假设最后一个事件持续到数据末尾

    event_durations = (df.index[idx_end_event] - df.index[idx_start_event]).total_seconds() / 60

    # 根据持续时间筛选有效事件
    valid_events_mask = event_durations > thre_time
    valid_event_durations = event_durations[valid_events_mask]
    valid_start_idx, valid_end_idx = idx_start_event[valid_events_mask], idx_end_event[valid_events_mask]

    events_all = [df.iloc[start:end + 1] for start, end in zip(valid_start_idx, valid_end_idx)]

    return events_all, valid_event_durations

def extract_monotype_events(events_all, thre_val=1, param_str='curnt_B'):
    """
    将所有有效事件分为“单一类型事件”和其他事件。
    """

    monotype_events = []
    other_events = []
    idx_monotype = np.zeros(len(events_all), dtype=bool)

    for i, event_ in enumerate(events_all):
        signal = event_[param_str].values

        # 使用方差来判断事件类型
        if np.var(signal) < thre_val:
            monotype_events.append(event_)
            idx_monotype[i] = True
        else:
            other_events.append(event_)

    return monotype_events, idx_monotype, other_events

def estimate_total_power(event):
    """
    估计泵的总功率，包括有功功率和无功功率。
    """

    # 电压 [V]
    v_A = event['volt_A'].values
    v_B = event['volt_B'].values
    v_C = event['volt_C'].values

    # 电流 [A]
    c_B = event['curnt_B'].values
    c_C = event['curnt_C'].values

    # 功率因数 [-]
    pf_A = event['factor_A'].values
    pf_B = event['factor_B'].values
    pf_C = event['factor_C'].values

    ave_pf = np.mean([pf_A, pf_B, pf_C], axis=0)
    ave_curnt = np.mean([c_B, c_C], axis=0)
    ave_volt = np.mean([v_A, v_B, v_C], axis=0)

    realP_proxy = 3 * ave_pf * ave_curnt * ave_volt / 1000
    reactP_proxy = 3 * np.sqrt(1 - np.square(ave_pf)) * ave_curnt * ave_volt / 1000

    event = event.assign(realP_proxy=realP_proxy, reactP_proxy=reactP_proxy)

    return event

