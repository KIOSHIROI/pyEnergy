import os
import pandas as pd
from pyEnergy import CONST
from datetime import datetime

class TimeSeriesReconstructor:
    def __init__(self, param_per_c):
        self.param_per_c = param_per_c

    def create_time_series(self, events_df, time_range):
        """
        根据事件记录生成时间序列信号
        Args:
            events_df: 事件记录DataFrame
            time_range: 时间范围(Tuple[datetime, datetime])
        Returns:
            pd.Series: 时间序列信号
        """
        time_index = pd.date_range(start=time_range[0], end=time_range[1], freq='T')
        series = pd.Series(0.0, index=time_index, name='signal')

        for _, event in events_df.iterrows():
            cluster_param = self.param_per_c[event['clusterId']]
            event_mask = (time_index >= event['starttime']) & (time_index <= event['endtime'])
            series.loc[event_mask] += cluster_param

        return series

    def process_cluster_file(self, input_path, output_dir):
        """
        处理单个聚类文件
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
        """
        # 读取事件数据
        events_df = pd.read_csv(input_path, parse_dates=['starttime', 'endtime'])
        
        # 获取时间范围
        min_time = events_df['starttime'].min()
        max_time = events_df['endtime'].max()
        
        # 生成时间序列
        ts = self.create_time_series(events_df, (min_time, max_time))
        
        # 保存结果
        filename = os.path.basename(input_path).replace('events', 'series')
        output_path = os.path.join(output_dir, filename)
        ts.to_csv(output_path, header=True)
        
        print(f'Generated: {output_path}')


def reconstruct_all_signals(param_per_c, model_name):
    """
    批量重建所有聚类信号
    Args:
        param_per_c: 各聚类参数值字典 {clusterId: param_value}
    """
    base_dir = os.path.join('output', model_name)
    input_dir = os.path.join(base_dir, 'signals')
    output_dir = os.path.join(base_dir, 'series')
    
    os.makedirs(output_dir, exist_ok=True)
    
    reconstructor = TimeSeriesReconstructor(param_per_c)
    
    for filename in os.listdir(input_dir):
        if filename.startswith('events_cluster'):
            input_path = os.path.join(input_dir, filename)
            reconstructor.process_cluster_file(input_path, output_dir)