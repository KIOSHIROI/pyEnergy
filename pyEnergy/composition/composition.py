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
    cluster_num = composer.param_per_c.shape[0]
    error  = []
    df_pred = [pd.DataFrame({'UTC Time':[], 'workingPower':[]}) for i in range(cluster_num)]

    start_idx = params.get('start_idx', 0)
    end_idx = params.get('end_idx', max_num)
    assert end_idx <= max_num
    plot = params.get('plot', False)
    total_start = time.time()

    for i in range(start_idx,end_idx):
        start = time.time()
        _, err = composer.compose(index=i)
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
        minute = period // 60
        second = period % 60
        if minute == 0:
            print(f"--{i+1}/{max_num}--err:{err:.3f}--time:{second:.3f}s--")
        else:
            print(f"--{i+1}/{max_num}--err:{err:.3f}--time:{minute}m{second:.3f}s--")
    total_end = time.time()
    mean_err = np.mean(error)
    period = total_end - total_start
    minute = period // 60
    second = period % 60
    if minute == 0:
        print(f"==total:{max_num}==total me:{mean_err:.3f}==total time:{second:.3f}s==")
    else:
        print(f"======total:{max_num}==total me:{mean_err:.3f}==total time:{minute}m{second:.3f}s======")

    # 确保目标目录存在
    output_dir = os.path.dirname(output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 递归创建目录

    # 写入错误文件
    with open(output_prefix + "_error.csv", "w+", encoding='utf-8') as f:
        f.write("event_no, mean_error,\n")
        for i, err in enumerate(error):
            f.write(f"{i}, {err},\n")

    # 写入预测信号文件
    for i in range(cluster_num):
        df_pred[i].to_csv(output_prefix + f"_signal{i+1}of{cluster_num}.csv")
        
        
class Composer():
    def __init__(self, fool, y_pred=None, **params):
        self.fool = fool
        self.reducer = None
        self.skip = False
        if y_pred is not None:
            self.fool.feature_backup["Cluster"] = y_pred
        self.param = params.get("param", None)
        print('composer init.')
        
    def set_param(self, param, fit=True, **params):
        self.param = param
        feature_param = CONST.param_feature_dict[self.param][1]
        
        # 获取每个簇的参数均值和簇大小
        clusters = self.fool.feature_backup.groupby('Cluster')[feature_param].agg(['mean', 'size'])
        self.param_per_c = clusters['mean'].values
        cluster_sizes = clusters['size'].values
        print("Initial cluster means:", self.param_per_c)
        print("-" * 10)
        
        if self.param == "realP_B":
            self.param_per_c /= 3
        
        if fit:
            print("fit=True")
            thres = params.get("threshold", 3 if self.param != "realP_B" else 1)

            # 进行簇合并迭代
            while len(self.param_per_c) > 1:
                print("Number of clusters:", len(self.param_per_c))
                
                # 计算所有簇之间的距离
                dif = np.abs(self.param_per_c[:, np.newaxis] - self.param_per_c)
                np.fill_diagonal(dif, np.inf)
                
                # 找到最小距离及其索引
                min_dist = np.min(dif)
                idx1, idx2 = np.unravel_index(np.argmin(dif), dif.shape)
                
                if min_dist < thres:
                    print(f"Merging clusters {idx1} and {idx2} with distance {min_dist}")
                    
                    # 根据簇内点的数量对两个簇进行加权平均
                    total_size = cluster_sizes[idx1] + cluster_sizes[idx2]
                    new_mean = (self.param_per_c[idx1] * cluster_sizes[idx1] + 
                                self.param_per_c[idx2] * cluster_sizes[idx2]) / total_size
                    
                    # 更新簇均值和簇大小
                    self.param_per_c[idx1] = new_mean
                    cluster_sizes[idx1] = total_size
                    
                    # 删除合并后的第二个簇
                    self.param_per_c = np.delete(self.param_per_c, idx2, axis=0)
                    cluster_sizes = np.delete(cluster_sizes, idx2, axis=0)
                else:
                    break
                if self.param_per_c.shape[0] < 2:
                    self.skip = True
        print("Final cluster means:", self.param_per_c)
        print("-" * 10)
        return self

    
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
        sols = sols.T
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

def compos(realP_perCluster, signal_reduced, low_bound=0, up_bound=6):
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