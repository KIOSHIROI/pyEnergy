# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:03:31 2018

@author: YuLi
"""

import numpy as np
from datetime import datetime
from glob     import glob
from ntpath   import basename 
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn import cluster
from sklearn.linear_model import OrthogonalMatchingPursuit

#def calc_std_prc(inArray, prc_value):
#
#    nr_elements  = len(inArray)
#    inArray_sort = np.sorted(inArray)
#
#    nr_ele_selected = math.floor( nr_elements * prc_value/100 )
#
#    samples = inArray_sort[0: nr_ele_selected]
#    std_prc = np.std(samples)
#
#    return std_prc

def calc_dominant_bin(inArray, bin_interval):
    """
    计算给定数组的主导区间（出现频率最高的区间）。

    参数:
    inArray (np.array): 输入的数值数据数组。
    bin_interval (int或数组形式): 区间的数量或各区间的边界。

    返回:
    float: 主导区间的中间值。
    """

    # 计算直方图
    hist, bins = np.histogram(inArray, bins=bin_interval)

    # 查找计数最高的区间的索引
    idx = hist.argmax()

    # 计算主导区间的中点
    dominant_bin = (bins[idx] + bins[idx + 1]) / 2

    return dominant_bin

def check_dateParser(inFile):
# check and determine the correct date parse for the input data

    tmp = np.genfromtxt(inFile,
                         max_rows    = 1,
                         usecols     = 0,
                         delimiter   = ',',
                         skip_header = 1,
                         dtype       = str
                         )

    try:
        datetime.strptime(str(tmp), '%Y-%m-%d %H:%M:%S')
        dateParser_str = '%Y-%m-%d %H:%M:%S'
    except ValueError:
        dateParser_str = '%d.%m.%y %H:%M'

    return dateParser_str


def check_nr_transformer(path_to_data):
# determine the number of different transformers, and group the corresponding input files
    
    data_files = glob(path_to_data + "/*.csv")
    
    transformer = []
    for each_file in data_files:
        tmp = basename(each_file).split('-')
        transformer.append( tmp[0] + '-' + tmp[1] )
    
    transformer_unique = np.unique(transformer)
    
    transformer_dict = {}
    for each_transformer in transformer_unique:
        files = glob( path_to_data + "/" + each_transformer + "*.csv" )
        transformer_dict[each_transformer] = files 
        
    return transformer_dict



def import_transformerData(data_files):
   
    selected_cols = range(0, 26)
    param_names = ['T', 
        'volt_A',   'volt_B',   'volt_C',   'volt_AB', 'volt_CA', 'volt_BC', 
        'curnt_A',  'curnt_B',  'curnt_C',  
        'realP_A',  'realP_B',  'realP_C',  'realP_tot', 
        'reacP_A',  'reacP_B',  'reacP_C',  'reacP_tot', 
        'aprntP_A', 'aprntP_B', 'aprntP_C', 'aprntP_tot', 
        'factor_A', 'factor_B', 'factor_C', 'factor_tot']

    list_ = []
    for file_ in data_files:
        
        date_parser_str = check_dateParser(file_)
        date_format_ = lambda x: pd.to_datetime(x, format=date_parser_str)
        
        df_tmp = pd.read_csv( file_,
                    delimiter   = ",",
                    header      = 1,
                    names       = param_names,
                    index_col   = 'T',
                    parse_dates = True,
                    date_format = date_format_,
                    usecols     = selected_cols # ( datetime, real power [W], reactive power [W] )
                )
        list_.append(df_tmp)
    
    
    df = pd.concat(list_)
    idx_duplicate = df.index.duplicated(keep='first')
    df = df[~idx_duplicate].sort_index()    
    
    # df.iloc[:,9:21]  = df.iloc[:,9:21]/1000    # change [W] into [kW]
    # df.iloc[:,21:25] = abs( df.iloc[:,21:25] ) # change power factor into positive values
    df.loc[:, 'realP_A':'aprntP_tot'] /= 1000  # change [W] into [kW]
    df.loc[:, 'factor_A':'factor_tot'] = abs(df.loc[:, 'factor_A':'factor_tot'])  # change power factor into positive values

    return df


def find_allEvents(df, thre_val, thre_time):
#    get events based on real power use.
    
    param_str = 'curnt_B'
    signals = df[param_str]
    idx_invalid = signals < thre_val
    signals[idx_invalid] = 0
 
    idx_valid = (signals > 0).values
    idx_valid = np.insert(1*idx_valid, 0, 0)
    
    idx_start_event = np.where( np.diff( idx_valid ) ==  1 )[0]
    idx_end_event   = np.where( np.diff( idx_valid ) == -1 )[0]-1 # minus 1 to get the idx of last record before the pump is turned off

    if len(idx_start_event) > len(idx_end_event):
        idx_start_event = idx_start_event[0:len(idx_end_event)]
    
    event_duration = []
    
    for idx_on, idx_off in zip( idx_start_event, idx_end_event ):
        event_duration.append(
            ( df.index[idx_off] - df.index[idx_on] ).total_seconds()/60
        )
    
    event_duration = np.array( event_duration, dtype='float')
    
    # identify valid event according to pumping duration thresehold
    idx_valid = np.where( event_duration > thre_time )
    t1 = idx_start_event[idx_valid]
    t2 = idx_end_event[idx_valid]
    event_duration_valid = event_duration[idx_valid]
    
    # collect all valid events into a list. Each element is a pandas dataframe that holds all parameters
    events_all = []
    
    for idx_on, idx_off in zip( t1, t2):
        events_all.append( df[idx_on:idx_off+1] )
        
    return (event_duration_valid, events_all)



def find_monotypeEvents(events_all, thre_val):
# collect all valid events into a list. Each element is a pandas dataframe that holds all parameters
    
    param_str = 'curnt_B' 
    
    monotypeEvents = []
    otherEvents    = []
    idx_monotype = np.zeros(len(events_all), dtype=bool)
    
    ctr = 0
    for event_ in events_all:
        signal = event_[param_str].values
        
        if ( np.var(signal) < thre_val ):
            monotypeEvents.append( event_ )
            idx_monotype[ctr] = True
        else:
            otherEvents.append( event_ )
        
        ctr +=1
        
    return ( monotypeEvents, idx_monotype, otherEvents )


def estimate_tot_power( event ):
    
    # voltage [V]
    v_A = event['volt_A'].values
    v_B = event['volt_B'].values
    v_C = event['volt_C'].values
    
    # current [A]
    c_B = event['curnt_B'].values
    c_C = event['curnt_C'].values
    
    # power factor [-]
    pf_A = event['factor_A'].values
    pf_B = event['factor_B'].values
    pf_C = event['factor_C'].values
    
    # estimate pump's total power 
    ave_pf    = np.mean(np.vstack( (pf_A, pf_B, pf_C)), axis = 0)
    ave_curnt = np.mean(np.vstack( (c_B, c_C)), axis = 0)
    ave_volt  = np.mean(np.vstack( (v_A, v_B, v_C)), axis = 0)
    
    realP_proxy  = 3* ave_pf * ave_curnt * ave_volt / 1000;
    reactP_proxy = 3* np.sqrt(1- np.power(ave_pf, 2)) * ave_curnt * ave_volt / 1000;

    event = event.assign( realP_proxy = realP_proxy, 
                          reactP_proxy = reactP_proxy )
    
    return event



def compute_features(monotype_events):
    
    feature_info = [ 
      'std. real power(ss)', 'ave. real power(ss)', 'max. real power(tr)',
      'std. reactive power(ss)', 'ave. reactive power(ss)', 'max. reactive power(tr)',
      'std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)'
    ]
    
    tr_steps = 5
    num_events = len(monotype_events)
    
    idx_used_events = np.ones(num_events, dtype=bool)
    feature_list = pd.DataFrame( 
            index   = range(0, num_events), 
            columns = feature_info
        )
    
    ctr = 0
    for event_ in monotype_events:
        
        # first tr_steps signals are considered as transient signals
        tr_realP  = event_['realP_proxy'][0:tr_steps]
        tr_reactP = event_['reactP_proxy'][0:tr_steps]
        tr_curntP = event_['curnt_B'][0:tr_steps]
        
        # the rest are considered as steady-state signals
        ss_realP  = event_['realP_proxy'][tr_steps:]
        ss_reactP = event_['reactP_proxy'][tr_steps:]
        ss_curntP = event_['curnt_B'][tr_steps:]
        
        # if sample number is less than 10, do not use this event to compute features;
        # TODO: if too much records is missing, do not use this event
        if len(ss_realP) < 10:
            idx_used_events[ctr] = False
            ctr += 1
            continue
   
        # compute each feature 
        feature_list[ feature_info[0] ][ctr] = np.std(ss_realP)
        feature_list[ feature_info[1] ][ctr] = np.mean(ss_realP)
        feature_list[ feature_info[2] ][ctr] = np.max(tr_realP)
        feature_list[ feature_info[3] ][ctr] = np.std(ss_reactP)
        feature_list[ feature_info[4] ][ctr] = np.mean(ss_reactP)
        feature_list[ feature_info[5] ][ctr] = np.max(tr_reactP)
        feature_list[ feature_info[6] ][ctr] = np.std(ss_curntP)
        feature_list[ feature_info[7] ][ctr] = np.mean(ss_curntP)
        feature_list[ feature_info[8] ][ctr] = np.max(tr_curntP)
             
        ctr += 1
        
    feature_list = feature_list.loc[idx_used_events, :]
    
    return (feature_list, idx_used_events)
   

def compute_score(feature_val, grps, w):
    
    scores = silhouette_samples(feature_val, grps)
    
    std_tmp = np.zeros( len(np.unique(grps)) )
    for i in np.unique(grps):
        idx = np.where(grps == i)
        std_tmp[i] = np.std(scores[idx])
        
    S_coeff   = np.mean(scores)
    score_std = -1*np.mean(std_tmp) #  minus to indicate a maximization since the total score is to be maximized
    
    perf_val = w[0]*S_coeff + w[1]*score_std
    
    return perf_val


def KMeans_elbow(feature_list, normalize = False, max_iter = None, repeats = 5):

    if (normalize == True):
        max_val = np.max(feature_list, axis = 0)
        min_val = np.min(feature_list, axis = 0)
        feature_list = (feature_list - min_val) / (max_val - min_val)
    
    if (max_iter == None):
        max_iter = np.ceil( np.sqrt(len(feature_list)) ).astype(int)
        
    ctr = 2
    weights = [0.5, 0.5]
    
    scores_final = np.zeros(max_iter)
    scores_final[0] = -1*np.inf
    
    grp_opt = []
    grp_opt.append(np.nan)
    
    while (ctr <= max_iter):
        
        grp_opt_tmp  = cluster.KMeans(n_clusters = ctr, n_init='auto').fit_predict(feature_list)
        perf_val_opt = compute_score(feature_list, grp_opt_tmp, weights)
        
        for cc in range(0, repeats): 
            grp_tmp = cluster.KMeans(n_clusters = ctr, n_init='auto').fit_predict(feature_list)
            perf_val_tmp = compute_score(feature_list, grp_tmp, weights)
                
            if (perf_val_tmp > perf_val_opt):
                grp_opt_tmp  = grp_tmp
                perf_val_opt = perf_val_tmp        
        
        scores_final[ctr-1] = perf_val_opt
        grp_opt.append( grp_opt_tmp )
        
        ctr += 1
                
    nrCluster_opt = np.argmax(scores_final)
    grp_opt_final = grp_opt[nrCluster_opt-1]
    
    scores_opt = compute_score(feature_list, grp_opt_final, weights)
    
    return (grp_opt_final, scores_opt)
   