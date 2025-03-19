import time
import numpy as np
import pandas as pd
from pyEnergy import CONST, drawer
from pyEnergy.composition.reducer import reduction
import os  

def auto_compose(composer, output_prefix, **params):
    '''
    params: (start_idx, end_idx, plot)
    '''
    max_num = len(composer.fool.other_event)
    cluster_num = len(composer.param_per_c)
    error  = []
    df_pred = [pd.DataFrame({'UTC Time':[], 'workingPower':[]}) for i in range(cluster_num)]

    start_idx = params.get('start_idx', 0)
    end_idx = params.get('end_idx', max_num)
    assert end_idx <= max_num
    plot = params.get('plot', False)
    total_start = time.time()

    composer.split_blocks(threshold=params.get('threshold', 3))
    composer.set_param(param=params.get("param", "realP_B"), fit=params.get("fit", True))
    composer.compos()
    for i in range(start_idx, end_idx):
        start = time.time()
        if i < len(composer.events):
            event = composer.events[i]
            composer.signal = composer.fool.feature_backup.loc[event[0]:event[1], composer.param]
            composer.x_values = composer.signal.index
            _, err = composer.compose()  
        if plot:
            composer.plot()
        err = np.mean(err)
        error.append(err)
        for j in range(cluster_num):
            x = composer.x_values
            signal = composer.pred_signal[j]
            signal = pd.DataFrame(zip(x, signal), columns=["UTC Time", "workingPower"])
            df_pred[j] = pd.concat([df_pred[j], signal]).drop_duplicates(subset='UTC Time')
        end = time.time()
        period = end - start
        print(f"--{i+1}/{max_num}--err:{err:.3f}--time:{period:.3f}s--")

    total_end = time.time()
    mean_err = np.mean(error)
    print(f"---total:{max_num}--total mean err:{mean_err:.3f}--total time:{total_end-total_start:.3f}s---")

    # 确保目标目录存在
    output_dir = os.path.dirname(output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 递归创建目录

    # 写入错误文件
    with open(output_prefix + "error.csv", "w+", encoding='utf-8') as f:
        f.write("event_no, mean_error,\n")
        for i, err in enumerate(error):
            f.write(f"{i}, {err},\n")

    # 写入预测信号文件
    for i in range(cluster_num):
        df_pred[i].to_csv(output_prefix + f"signal{i+1}of{cluster_num}.csv")
        
        
class Composer():
    def __init__(self, fool, y_pred=None, **params):
        self.fool = fool
        self.reducer = None
        self.skip = False
        if y_pred is not None:
            self.fool.feature_backup["Cluster"] = y_pred
        self.param = params.get("param", None)
        print('composer init.')

    def split_blocks(self,threshold=3):
        self.events=[]
        df=self.fool.feature_backup
        if df.empty:
            return
        s = df.index[0]
        partition_events = []
        power_feature = self.fool.feature_backup.loc[:, CONST.feature_info[1]].to_numpy()
        for i in range(1, len(df)):
            print(i, len(df) )
            current = power_feature[i]
            prev = power_feature[i-1]
            if abs(current - prev)>threshold:
                e = df.index[i]
                partition_events.append((s, e))
                s = e
        if s<df.index[-1]:
            partition_events.append((s,df.index[-1]))
        self.events = partition_events

    def set_param(self, param, fit=True, **params):
        self.param = param
        feature_param = CONST.param_feature_dict[self.param][1]
        self.param_per_c = []
        current_signals = []
        for (s, e) in self.events:
            event_data = self.fool.feature_backup[s:e]
            event_mean = event_data[feature_param].mean()
            self.param_per_c.append(event_mean)
            if len(current_signals) == 0:
                current_signals.append(event_mean)
        self.current_signals = current_signals


    def compos(self):
        from scipy.spatial.distance import cdist
        count_signals = {}
        trend_history = []
        current_signals = self.current_signals.copy()
        param_per_c_padded = np.array(self.param_per_c).reshape(-1, 1) 
        
        for i in range(1, len(param_per_c_padded)):
            prev_param = param_per_c_padded[i - 1]
            curr_param = param_per_c_padded[i]
            delta = curr_param - prev_param
            if delta > 0:
                distances = cdist([curr_param], param_per_c_padded, 'euclidean')
                n = np.argmin(distances)
                count_signals[n] = count_signals.get(n, 0) + 1
                current_signals.append(curr_param)
            elif delta < 0:
                valid_signals = [s for s in current_signals if s > 0]
                if valid_signals:
                    distances = cdist([curr_param], np.array(valid_signals).reshape(-1, 1), 'euclidean')
                    n = np.argmin(distances)
                    count_signals[n] = count_signals.get(n, 0) - 1
                    current_signals.pop(n)
            trend_history.append(count_signals.copy())
    
        self.trend_changes = trend_history

    def set_reducer(self, reducer, reducer_params={}):
        self.reducer = reduction(reducer)(**reducer_params)
        self.reducer_params = reducer_params
        return self

    def compose(self,index=0):

        other_events = self.fool.other_event
        idx = index
        event = other_events[idx]
        self.signal = event[self.param]
        self.x_values = self.signal.index
        if self.reducer is not None:
            self.signal, _ = self.reducer.reduce(signal=self.signal, **self.reducer_params)
        self.sols, errors = compos(self.param_per_c, self.signal)
        self.get_pred_signal()
        return self.sols, errors
    
    def get_pred_signal(self):
        n_clusters = len(self.sols[0])
        sols = np.array(self.sols)
        sols = sols * np.array(self.param_per_c.reshape(1,n_clusters))
        self.pred_signal = sols
    
    def plot(self, plot=True, save_path=None):
        signal = reconstruct_signal(self.sols, self.param_per_c)
        drawer.draw_result(self.signal, signal, self.sols, self.param_per_c, plot=plot, save=save_path, x_values=self.x_values)


def reconstruct_signal(sols, phaseB_perCluster):
    reconstructed = []
    for sol in sols:
        reconstructed_signal = sum(phaseB_perCluster[i] * sol[i] for i in range(len(sol)))
        reconstructed.append(reconstructed_signal)
    return np.array(reconstructed)
