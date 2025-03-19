import time
import numpy as np
import pandas as pd
from pyEnergy import CONST, drawer
from pyEnergy.composition.reducer import reduction
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
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

    composer.split_events(threshold=params.get('threshold',3))
    composer.set_param(param=params.get("param", "default"))
    composer.merge_events()
    for i in range(start_idx,end_idx):
        start = time.time()
        if i<len(composer.events):
            event = composer.events[i]
            _, err = composer.compose(event)
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

    def split_events(self,threshold=3):
        self.events=[]
        df=self.fool.feature_backup
        if df.empty:
            return
        s = df.index[0]
        partition_events = []

        for i in range(1, len(df)):
            current=df['Power'][i]
            prev=df['Power'][i-1]
            if abs(current-prev)>threshold:
                e=df.index[i]
                partition_events.append((s, e))
                s=e
        if s<df.index[-1]:
            partition_events.append((s,df.index[-1]))
        self.events = partition_events

    def set_param(self, param, fit=True, **params):
        self.param = param
        feature_param = CONST.param_feature_dict[self.param][1]
        self.param_per_c=[]
        current_signals=[]
        for (s,e) in self.events:
            event_data = self.fool.feature_backup[s:e]
            clusters=event_data.groupby('Cluster')[feature_param].mean()
            self.param_per_c.append(clusters.values)
            if len(current_signals) == 0:
                current_signals.append(clusters.values)
        self.current_signals = current_signals


    def merge_events(self):
        from scipy.spatial.distance import cdist
        count_signals={}
        trend_history=[]
        current_signals=self.current_signals.copy()
        for i in range(1, len(self.param_per_c)):
            prev_param=self.param_per_c[i - 1]
            curr_param=self.param_per_c[i]
            delta=curr_param.mean()-prev_param.mean()
            if delta>0:
                distances=cdist([curr_param], self.param_per_c, 'euclidean')
                n=np.argmin(distances)
                count_signals[n] = count_signals.get(n, 0) + 1
                current_signals.append(curr_param)
            elif delta<0:
                valid_signals=[s for s in current_signals if s.mean()>0]
                if valid_signals:
                    distances=cdist([curr_param], valid_signals, 'euclidean')
                    n=np.argmin(distances)
                    count_signals[n]=count_signals.get(n, 0)-1
                    current_signals.pop(n)
            trend_history.append(count_signals.copy())

        self.trend_changes=trend_history

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


from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value
import numpy as np
from joblib import Parallel, delayed  # 用于并行处理

def compos(realP_perCluster, signal_reduced, low_bound=0, up_bound=8):
    sols = []
    errors = []
    n_clusters = len(realP_perCluster)
    
    # 预先计算求和表达式
    sum_realP_perCluster = np.sum(realP_perCluster, axis=0)
    
    # 并行处理信号
    results = Parallel(n_jobs=-1)(delayed(process_signal)(realP_perCluster, signal, sum_realP_perCluster, low_bound, up_bound) for signal in signal_reduced)
    
    for result in results:
        sols.append(result[0])
        errors.append(result[1])
    
    return sols, errors

def process_signal(realP_perCluster, signal, sum_realP_perCluster, low_bound, up_bound):
    prob = LpProblem("Signal_Decomposition", LpMinimize)
    
    # 定义优化变量
    x = LpVariable.dicts("x", range(len(realP_perCluster)), lowBound=low_bound, upBound=up_bound, cat='Integer')
    error = LpVariable('error', lowBound=0)
    
    # 目标函数：最小化误差和 max_x
    prob += error - lpSum(x)
    
    # 添加约束：线性组合的值必须等于信号加上误差
    prob += lpSum(realP_perCluster[j] * x[j] for j in range(len(realP_perCluster))) + error >= signal
    prob += lpSum(realP_perCluster[j] * x[j] for j in range(len(realP_perCluster))) - error <= signal

    # 求解问题
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # 保存最优解和误差
    solution = [int(value(x[i])) for i in range(len(realP_perCluster))]
    error_value = value(error)
    return solution, error_value