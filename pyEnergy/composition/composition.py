import time
import numpy as np
import pandas as pd
from pyEnergy import CONST, drawer
from pyEnergy.composition.reducer import reduction
import os  
from scipy.spatial.distance import cdist

def auto_compose(composer, output_prefix, **params):
    '''
    自动执行负荷分解过程，处理多个事件并保存结果，包括启动和关闭事件
    
    自动执行负荷分解过程，处理多个事件并保存结果
    
    Args:
        composer: Composer对象，用于执行负荷分解
        output_prefix: str，输出文件的前缀路径
        **params: 可选参数
            - start_idx: int，起始事件索引，默认为0
            - end_idx: int，结束事件索引，默认为最大事件数
            - plot: bool，是否绘制结果图，默认为False
            - threshold: float，分块阈值，默认为3
            - param: str，使用的参数名，默认为"realP_B"
            - fit: bool，是否进行拟合，默认为True
    '''
    # 确保输出目录存在
    output_dir = os.path.dirname(output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    max_num = len(composer.fool.other_event) # 需要分解的事件的总数
    cluster_num = len(composer.param_per_c) # 聚类数量（机井数量）
    error  = []
    # df_pred = [pd.DataFrame({'UTC Time':[], 'workingPower':[]}) for i in range(cluster_num)] # 预测信号

    start_idx = params.get('start_idx', 0)
    end_idx = params.get('end_idx', max_num)
    assert end_idx <= max_num
    plot = params.get('plot', False)

    print(f"开始负荷分解，输出文件前缀: {output_prefix}")
    total_start = time.time()

    for i in range(start_idx, end_idx):
        start = time.time()
        composer.split_blocks(index=i, threshold=params.get('threshold', 10)) # 分块
        composer.compos(index=i)
        end = time.time()
        period = end - start
        # print(f"--{i+1}/{max_num}--time:{period:.3f}s--")
    composer.compos_monotype()
    total_end = time.time()
    # mean_err = np.mean(error)
    print(f"---total time:{total_end-total_start:.3f}s---")

    # 保存启动和关闭事件
    events_df = pd.DataFrame(composer.signal_events, columns=['recordId', 'clusterId', 'eventId', 'starttime', 'endtime', 'aggNum'])
    events_df.to_csv(output_prefix + "_events.csv", index=False)

        
        
class Composer():
    '''
    负荷分解器类，用于执行电力负荷的分解和重构
    
    该类实现了基于聚类结果的负荷分解算法，可以将复杂的电力负荷信号分解为多个基本负荷组件
    '''
    
    def __init__(self, fool, y_pred=None, **params):
        '''
        初始化Composer对象
        
        Args:
            fool: Fool对象，包含特征数据和事件信息
            y_pred: array-like，聚类预测结果，默认为None
            **params: 可选参数
                - param: str，使用的参数名，默认为None
        '''
        self.fool = fool
        self.reducer = None
        self.skip = False
        self.signal_events = []
        self.current_signals = []
        self.record_counter = 0
        if y_pred is not None:
            self.clusters = y_pred
            # 将聚类结果添加到feature_backup中
            self.fool.feature_backup['Cluster'] = y_pred
        self.param = params.get("param", None)
        print('composer init.')

    def set_param(self, param, fit=True, **params):
        '''
        设置分解参数和计算每个聚类的参数均值
        
        Args:
            param: str，要使用的参数名
            fit: bool，是否进行拟合，默认为True
            **params: 其他参数
        '''
        self.param = param
        feature_param = CONST.param_feature_dict[self.param][1]
        clusters = self.fool.feature_backup.groupby('Cluster')[feature_param].agg(['mean', 'size'])
        self.param_per_c = clusters['mean'].values


    def split_blocks(self, index=0, threshold=3, window_size=5):
        '''根据索引处理单个事件并进行分块
        
        Args:
            index: int，事件索引，默认为0
            threshold: float，分块阈值，默认为3
            window_size: int，平滑窗口大小，默认为5
        '''
        other_events = self.fool.other_event
        if not other_events or index >= len(other_events):
            raise ValueError(f"No event data available at index {index}")
        event = other_events[index]
        # 获取事件的时间范围
        event_times = pd.to_datetime(event.index)
        # 获取feature_backup的时间范围
        feature_times = pd.to_datetime(self.fool.original_data.index)
        # 找到最接近的时间点
        start_idx = feature_times.searchsorted(event_times[0])
        end_idx = feature_times.searchsorted(event_times[-1])
        # 获取对应的数据
        df = self.fool.original_data.iloc[start_idx:end_idx+1]
        if df.empty:
            raise ValueError(f"No data found for event at index {index}")
        
        s = df.index[0]
        partition_events = []
        # 对功率特征进行移动平均平滑处理
        power_series = df.loc[:, self.param]
        smoothed_power = power_series.rolling(window=window_size, center=True, min_periods=1).mean()
        power_feature = smoothed_power.to_numpy()
        for i in range(1, len(df)):
            current = power_feature[i]
            prev = power_feature[i-1]
            if abs(current - prev) > threshold:
                e = df.index[i]
                partition_events.append((s, e))
                s = e
        if s<df.index[-1]:
            partition_events.append((s,df.index[-1]))
        self.events = partition_events

    def compos(self, index=0):
        '''根据索引处理单个事件并进行分解
        
        Args:
            index: int，事件索引，默认为0
        '''
        
        # 初始化变量
        # active_wells: 维护当前活跃的机井列表，每个元素是(聚类索引, 功率值, 启动时间)的元组
        # signal_events: 记录启动和关闭事件，每个元素是(clusterId, eventId, starttime, endtime, aggNum)的元组
        active_wells = []
        count_signals = {}
        
        # 将每个聚类的参数值转换为列向量，用于计算欧氏距离
        param_per_c_padded = np.array(self.param_per_c).reshape(-1, 1)
        
        # 计算每个数据块的均值
        block_means = []
        for start, end in self.events:
            block_data = self.fool.original_data.loc[start:end, self.param]
            block_means.append(block_data.mean())
        
        # 分析相邻数据块的差值
        for i in range(len(block_means)-1):
            curr_mean = block_means[i]
            next_mean = block_means[i+1]
            delta = next_mean - curr_mean
            
            if delta > 0:  # 新增机井
                # 计算当前信号与各聚类中心的欧氏距离
                distances = cdist([[delta]], param_per_c_padded, 'euclidean')
                # 找到最近的聚类
                cluster_idx = np.argmin(distances)
                # 记录机井启动
                start_time = self.events[i][1]
                active_wells.append((cluster_idx, delta, start_time))
                # 更新信号计数
                count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) + 1
                
            elif delta < 0:  # 关闭机井
                if active_wells:
                    # 计算与已启动机井的距离
                    well_powers = np.array([w[1] for w in active_wells]).reshape(-1, 1)
                    distances = cdist([[abs(delta)]], well_powers, 'euclidean')
                    # 找到最近的机井
                    well_idx = np.argmin(distances)
                    cluster_idx, _, start_time = active_wells[well_idx]
                    # 更新信号计数
                    count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) - 1
                    # 记录事件
                    end_time = self.events[i][1]
                    agg_num = count_signals[cluster_idx]
                    self.record_counter += 1
                    self.signal_events.append((self.record_counter, cluster_idx, index, start_time, end_time, agg_num))
                    # 移除关闭的机井
                    active_wells.pop(well_idx)
            
            # 记录当前时间点的信号分布状态
        
        # 处理剩余活跃机井的关闭信号
        for cluster_idx, power, start_time in active_wells:
            count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) - 1
            # 记录事件
            end_time = self.events[-1][1]
            agg_num = count_signals[cluster_idx]
            self.record_counter += 1
            self.signal_events.append((self.record_counter, cluster_idx, index, start_time, end_time, agg_num))
    
    def compos_monotype(self, threshold=3, window_size=5):
        """处理所有事件并进行单一类型的分块分析
        
        Args:
            threshold: float，分块阈值，默认为3
            window_size: int，平滑窗口大小，默认为5
        """
        other_events = self.fool.other_event
        if not other_events:
            raise ValueError("No events available")
            
        for event_idx, event in enumerate(other_events):
            # 获取事件的时间范围
            event_times = pd.to_datetime(event.index)
            # 获取feature_backup的时间范围
            feature_times = pd.to_datetime(self.fool.original_data.index)
            # 找到最接近的时间点
            start_idx = feature_times.searchsorted(event_times[0])
            end_idx = feature_times.searchsorted(event_times[-1])
            # 获取对应的数据
            df = self.fool.original_data.iloc[start_idx:end_idx+1]
            if df.empty:
                continue
            
            s = df.index[0]
            partition_events = []
            # 对功率特征进行移动平均平滑处理
            power_series = df.loc[:, self.param]
            smoothed_power = power_series.rolling(window=window_size, center=True, min_periods=1).mean()
            power_feature = smoothed_power.to_numpy()
            
            # 分块处理
            for i in range(1, len(df)):
                current = power_feature[i]
                prev = power_feature[i-1]
                if abs(current - prev) > threshold:
                    e = df.index[i]
                    partition_events.append((s, e))
                    s = e
            if s < df.index[-1]:
                partition_events.append((s, df.index[-1]))
                
            # 如果没有分块，继续下一个事件
            if not partition_events:
                continue
                
            active_wells = []
            count_signals = {}
            
            # 将每个聚类的参数值转换为列向量，用于计算欧氏距离
            param_per_c_padded = np.array(self.param_per_c).reshape(-1, 1)
            
            # 计算每个数据块的均值
            block_means = []
            for start, end in partition_events:
                block_data = df.loc[start:end, self.param]
                block_means.append(block_data.mean())
            
            # 分析相邻数据块的差值
            for i in range(len(block_means)-1):
                curr_mean = block_means[i]
                next_mean = block_means[i+1]
                delta = next_mean - curr_mean
                
                if delta > 0:  # 新增机井
                    # 计算当前信号与各聚类中心的欧氏距离
                    distances = cdist([[delta]], param_per_c_padded, 'euclidean')
                    # 找到最近的聚类
                    cluster_idx = np.argmin(distances)
                    # 记录机井启动
                    start_time = partition_events[i][1]
                    active_wells.append((cluster_idx, delta, start_time))
                    # 更新信号计数
                    count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) + 1
                    
                elif delta < 0:  # 关闭机井
                    if active_wells:
                        # 计算与已启动机井的距离
                        well_powers = np.array([w[1] for w in active_wells]).reshape(-1, 1)
                        distances = cdist([[abs(delta)]], well_powers, 'euclidean')
                        # 找到最近的机井
                        well_idx = np.argmin(distances)
                        cluster_idx, _, start_time = active_wells[well_idx]
                        # 更新信号计数
                        count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) - 1
                        # 记录事件
                        end_time = partition_events[i][1]
                        agg_num = count_signals[cluster_idx]
                        self.record_counter += 1
                        self.signal_events.append((self.record_counter, cluster_idx, event_idx, start_time, end_time, agg_num))
                        # 移除关闭的机井
                        active_wells.pop(well_idx)
            
            # 处理剩余活跃机井的关闭信号
            for cluster_idx, power, start_time in active_wells:
                count_signals[cluster_idx] = count_signals.get(cluster_idx, 0) - 1
                # 记录事件
                end_time = partition_events[-1][1]
                agg_num = count_signals[cluster_idx]
                self.record_counter += 1
                self.signal_events.append((self.record_counter, cluster_idx, event_idx, start_time, end_time, agg_num))

    def set_reducer(self, reducer, reducer_params={}):
        '''
        设置信号降维器
        
        Args:
            reducer: str，降维器的名称
            reducer_params: dict，降维器的参数，默认为空字典
        
        Returns:
            self: 返回对象本身，支持链式调用
        '''
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
        self.sols, errors = self.compos(self.param_per_c, self.signal)
        self.get_pred_signal()
        return self.sols, errors
    
    def get_pred_signal(self):
        '''
        根据分解结果计算预测信号
        '''
        n_clusters = len(self.sols[0])
        sols = np.array(self.sols)
        sols = sols * np.array(self.param_per_c.reshape(1,n_clusters))
        self.pred_signal = sols
    
    def plot(self, plot=True, save_path=None):
        '''
        绘制分解结果
        
        Args:
            plot: bool，是否显示图形，默认为True
            save_path: str，保存路径，默认为None
        '''
        signal = reconstruct_signal(self.sols, self.param_per_c)
        drawer.draw_result(self.signal, signal, self.sols, self.param_per_c, plot=plot, save=save_path, x_values=self.x_values)


def reconstruct_signal(sols, phaseB_perCluster):
    '''
    重构信号
    
    Args:
        sols: array-like，分解结果
        phaseB_perCluster: array-like，每个聚类的参数值
    
    Returns:
        array: 重构后的信号
    '''
    reconstructed = []
    for sol in sols:
        reconstructed_signal = sum(phaseB_perCluster[i] * sol[i] for i in range(len(sol)))
        reconstructed.append(reconstructed_signal)
    return np.array(reconstructed)
