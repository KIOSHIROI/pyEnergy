import numpy as np
import pandas as pd
from pyEnergy.core import estimate_total_power

def compute_season(event_):
    event_month = np.argmax(np.bincount(event_.index.month.to_list()))
    if 2 <= event_month <= 6:
        return True
    return False

def compute_features(monotype_events, drought=False):
    monotype_events = [estimate_total_power(event_) for event_ in monotype_events]
    tr_steps = 5  # 暂态阶段的时间步数定义
    num_events = len(monotype_events)  # 事件总数

    idx_used_events = np.ones(num_events, dtype=bool)
    feature_list = pd.DataFrame(index=range(num_events))

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
        feature_list.loc[ctr, 'drought season'] = compute_season(event_)
    feature_list = feature_list[idx_used_events]
    if drought:
        idx_used_events = feature_list['drought season'] == 1
        feature_list = feature_list[idx_used_events]
        
    feature_list = feature_list.drop(['drought season'], axis=1)
    return feature_list, idx_used_events
