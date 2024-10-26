import numpy as np
import pandas as pd
def find_all_events(df, thre_val=3, thre_time=10,  param_str = 'curnt_B'):

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

def extract_monotype_events(events_all, thre_val=3, param_str='curnt_B'):
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


def compute_features(monotype_events, feature_info=None):
    monotype_events = [estimate_total_power(event_) for event_ in monotype_events]
    if feature_info is None:
        feature_info = [
            'std. real power(ss)', 'ave. real power(ss)', 'trend real power(ss)', 'max. real power(tr)',
            'std. reactive power(ss)', 'ave. reactive power(ss)', 'trend reactive power(ss)', 'max. reactive power(tr)',
            'std. phase B current(ss)', 'ave. phase B current(ss)', 'trend phase B current(ss)', 'max. phase B current(tr)'
        ]

    tr_steps = 5  # 暂态阶段的时间步数定义
    num_events = len(monotype_events)  # 事件总数

    idx_used_events = np.ones(num_events, dtype=bool)
    feature_list = pd.DataFrame(index=range(num_events), columns=feature_info)

    for ctr, event_ in enumerate(monotype_events):
        tr_realP = event_['realP_proxy'][:tr_steps]
        tr_reactP = event_['reactP_proxy'][:tr_steps]
        tr_curntB = event_['curnt_B'][:tr_steps]

        ss_realP = event_['realP_proxy'][tr_steps:]
        ss_reactP = event_['reactP_proxy'][tr_steps:]
        ss_curntB = event_['curnt_B'][tr_steps:]

        if len(ss_realP) < 10 or np.isnan(ss_realP).sum() > len(ss_realP) * 0.2:
            idx_used_events[ctr] = False
            continue

        # 将时间差转换为分钟
        time_num = pd.to_datetime(event_.index[tr_steps:])
        time_diff_minutes = np.diff(time_num) / np.timedelta64(1, 'm')  # 转换为分钟

        # 合并并计算累积和，包含初始时间步0
        time_steps = np.cumsum(np.concatenate(([0], time_diff_minutes)))

        # 计算稳态阶段的特征
        feature_list.loc[ctr, 'std. real power(ss)'] = np.nanstd(ss_realP)
        feature_list.loc[ctr, 'ave. real power(ss)'] = np.nanmean(ss_realP)
        feature_list.loc[ctr, 'trend real power(ss)'] = np.polyfit(time_steps, ss_realP, 1)[0]
        feature_list.loc[ctr, 'max. real power(tr)'] = np.nanmax(tr_realP)

        feature_list.loc[ctr, 'std. reactive power(ss)'] = np.nanstd(ss_reactP)
        feature_list.loc[ctr, 'ave. reactive power(ss)'] = np.nanmean(ss_reactP)
        feature_list.loc[ctr, 'trend reactive power(ss)'] = np.polyfit(time_steps, ss_reactP, 1)[0]
        feature_list.loc[ctr, 'max. reactive power(tr)'] = np.nanmax(tr_reactP)

        feature_list.loc[ctr, 'std. phase B current(ss)'] = np.nanstd(ss_curntB)
        feature_list.loc[ctr, 'ave. phase B current(ss)'] = np.nanmean(ss_curntB)
        feature_list.loc[ctr, 'trend phase B current(ss)'] = np.polyfit(time_steps, ss_curntB, 1)[0]
        feature_list.loc[ctr, 'max. phase B current(tr)'] = np.nanmax(tr_curntB)

    feature_list = feature_list[idx_used_events]

    return feature_list, idx_used_events
