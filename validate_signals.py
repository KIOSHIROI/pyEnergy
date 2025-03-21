import pandas as pd
import os
from datetime import datetime, timedelta

def validate_signal_events(validation_events_path, signals_dir):
    """验证signals文件夹下的分解结果与验证集的符合程度
    
    Args:
        validation_events_path: str, 验证集事件文件路径
        signals_dir: str, signals文件夹路径
        
    Returns:
        dict: 每个聚类的准确率评估结果
    """
    # 读取验证集事件
    validation_df = pd.read_csv(validation_events_path)
    validation_df['starttime'] = pd.to_datetime(validation_df['starttime'])
    validation_df['endtime'] = pd.to_datetime(validation_df['endtime'])
    
    # 存储每个聚类的评估结果
    cluster_accuracies = {}
    
    # 遍历signals文件夹下的所有聚类文件
    for filename in os.listdir(signals_dir):
        if filename.startswith('events_cluster_') and filename.endswith('.csv'):
            cluster_id = int(filename.split('_')[-1].split('.')[0])
            
            # 读取聚类事件数据
            cluster_df = pd.read_csv(os.path.join(signals_dir, filename))
            cluster_df['starttime'] = pd.to_datetime(cluster_df['starttime'])
            cluster_df['endtime'] = pd.to_datetime(cluster_df['endtime'])
            
            # 统计正确检测的事件数
            correct_starts = 0
            correct_ends = 0
            
            # 对每个验证集事件进行检查
            for _, val_event in validation_df.iterrows():
                # 检查启动信号
                start_window_start = val_event['starttime'] - timedelta(hours=1)
                start_window_end = val_event['starttime']
                
                # 检查是否在时间窗口内有启动信号
                start_detected = any((cluster_df['starttime'] >= start_window_start) & 
                                   (cluster_df['starttime'] <= start_window_end))
                if start_detected:
                    correct_starts += 1
                
                # 检查关闭信号
                end_window_start = val_event['endtime'] - timedelta(hours=2)
                end_window_end = val_event['endtime'] - timedelta(hours=1)
                
                # 检查是否在时间窗口内有关闭信号
                end_detected = any((cluster_df['endtime'] >= end_window_start) & 
                                 (cluster_df['endtime'] <= end_window_end))
                if end_detected:
                    correct_ends += 1
            
            # 计算准确率
            total_events = len(validation_df)
            start_accuracy = correct_starts / total_events
            end_accuracy = correct_ends / total_events
            avg_accuracy = (start_accuracy + end_accuracy) / 2
            
            # 存储结果
            cluster_accuracies[cluster_id] = {
                'start_accuracy': start_accuracy,
                'end_accuracy': end_accuracy,
                'avg_accuracy': avg_accuracy,
                'total_validation_events': total_events,
                'correct_starts': correct_starts,
                'correct_ends': correct_ends
            }
    
    return cluster_accuracies

def print_validation_results(accuracies):
    """打印验证结果
    
    Args:
        accuracies: dict, validate_signal_events返回的准确率结果
    """
    print("\n验证结果汇总:")
    print("-" * 60)
    print(f"{'聚类ID':^8} | {'启动准确率':^12} | {'关闭准确率':^12} | {'平均准确率':^12}")
    print("-" * 60)
    
    for cluster_id, metrics in sorted(accuracies.items()):
        print(f"{cluster_id:^8} | {metrics['start_accuracy']*100:^11.2f}% | "
              f"{metrics['end_accuracy']*100:^11.2f}% | {metrics['avg_accuracy']*100:^11.2f}%")
    
    print("-" * 60)
    
    # 计算并打印最高平均准确率的聚类信息
    best_cluster_id = max(accuracies, key=lambda x: accuracies[x]['avg_accuracy'])
    best_metrics = accuracies[best_cluster_id]
    print(f"\n最高平均准确率的聚类: {best_cluster_id}")
    print(f"启动准确率: {best_metrics['start_accuracy']*100:.2f}%")
    print(f"关闭准确率: {best_metrics['end_accuracy']*100:.2f}%")
    print(f"平均准确率: {best_metrics['avg_accuracy']*100:.2f}%")
    print(f"总验证事件数: {best_metrics['total_validation_events']}")
    print(f"正确检测启动事件数: {best_metrics['correct_starts']}")
    print(f"正确检测关闭事件数: {best_metrics['correct_ends']}")