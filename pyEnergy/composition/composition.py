import numpy as np
import pandas as pd
from pyEnergy import CONST, drawer
from pyEnergy.composition.reducer import reduction
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD

class Composer():
    def __init__(self, fool, y_pred=None, **params):
        self.fool = fool
        self.reducer = None
        if y_pred is not None:
            self.fool.feature_backup["Cluster"] = y_pred
        self.param = params.get("param", None)
        
    def set_param(self, param, fit=True):
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
            thres = 1 if self.param != "realP_B" else 0.5

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

        print("Final cluster means:", self.param_per_c)
        print("-" * 10)
        return self

    
    def set_reducer(self, reducer, reducer_params={}):
        self.reducer = reduction(reducer)(**reducer_params)
        self.reducer_params = reducer_params

    def compose(self,index=0, fit=True):
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


def compos(realP_perCluster, signal_reduced, low_bound=0, up_bound=2):
    sols = []
    errors = []  # 用于保存每个信号的误差
    n_clusters = len(realP_perCluster)  # 聚类数量
    
    for signal in signal_reduced:
        prob = LpProblem("Signal_Decomposition", LpMinimize)
        
        # 定义优化变量
        x = [LpVariable(f'x_{i}', lowBound=low_bound, upBound=up_bound, cat='Integer') for i in range(n_clusters)]
        error = LpVariable('error', lowBound=0)
        
        # 目标函数：最小化误差和 max_x
        prob += error - lpSum(x)
        
        # 添加约束：线性组合的值必须等于信号加上误差
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) + error >= signal
        prob += lpSum(realP_perCluster[j] * x[j] for j in range(n_clusters)) - error <= signal
    

        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # 保存最优解和误差
        sols.append([int(value(x[i])) for i in range(n_clusters)])
        errors.append(value(error))  # 保存误差值
    
    return sols, errors  # 返回最优解和误差值
