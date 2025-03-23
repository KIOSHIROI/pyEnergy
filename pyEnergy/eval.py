import os
import datetime
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pyEnergy.drawer as dw

def setup_logging():
    """设置日志记录"""
    log_dir = os.path.join('output', 'log')
    os.makedirs(log_dir, exist_ok=True)
    _ = os.path.join(log_dir, f'{datetime.datetime.now().strftime("%Y%m%d")}')
    os.makedirs(_, exist_ok=True)
    log_file = os.path.join(_, datetime.datetime.now().strftime("%H%M%S") + '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def compute_time_inclution(df_pred, df_val):
    """计算时间包含度"""
    diff = df_pred - df_val
    inclusion = 1 - (diff < 0).sum() / (df_val > 0).sum()
    return inclusion.iloc[0]

def compute_inclution(df_pred, df_val):
    """计算功率包含度"""
    diff = df_pred - df_val
    inclusion = 1 - df_val[diff < 0].sum() / df_val.sum()
    inclusion = inclusion.iloc[0]
    return inclusion

def compute_time_matching(df_pred, df_val):
    """计算时间匹配度：关注时间点的重叠匹配情况"""
    pred_active = df_pred > 0
    val_active = df_val > 0
    overlap = (pred_active & val_active).sum()
    total_active = (pred_active | val_active).sum()
    matching = overlap.iloc[0] / total_active.iloc[0] if total_active.iloc[0] > 0 else 0
    return matching

def compute_matching(df_pred, df_val):
    """计算功率匹配度：关注功率值的匹配情况"""
    diff = df_pred - df_val
    matching = (df_val.sum() - df_val[diff < 0].sum()) / df_pred.sum()
    return matching.iloc[0]

def compute_power_error_by_state(df_pred, df_val):
    """计算在有信号和无信号时段的功率误差

    Args:
        df_pred: 预测的功率数据
        df_val: 验证集的功率数据

    Returns:
        dict: 包含有信号和无信号时段的平均功率误差
    """
    power_error = abs(df_pred - df_val)
    signal_mask = df_val > 0
    active_error = power_error[signal_mask].mean().iloc[0] if signal_mask.any().iloc[0] else 0
    inactive_error = power_error[~signal_mask].mean().iloc[0] if (~signal_mask).any().iloc[0] else 0
    
    return {
        'active_power_error': active_error,
        'inactive_power_error': inactive_error
    }

def compute_mse(df_pred, df_val):
    """计算均方误差（MSE）"""
    diff = df_pred - df_val
    mse = np.mean(diff**2)
    return mse

def compute_infomation_ratio(df_pred, df_val):
    n_pred = df_pred.sum().iloc[0]
    n_val = df_val.sum().iloc[0]
    return n_val / n_pred

def compute_mae(df_pred, df_val):
    """计算平均绝对误差（MAE）"""
    diff = abs(df_pred - df_val)
    mae = np.mean(diff)
    return mae


def interpolation_pred(path, df_val):
    """对预测数据进行插值处理"""
    df_pred_1 = pd.read_csv(path)
    df_pred_1 = df_pred_1.drop(["Unnamed: 0"], axis=1)
    df_pred_1['UTC Time'] = pd.to_datetime(df_pred_1['UTC Time'])
    df_pred_1.set_index('UTC Time', inplace=True)
    df_pred_1 = df_pred_1.resample('T').interpolate(method='nearest')
    time_index = pd.date_range(start=min(df_pred_1.index.min(), df_val.index.min()),
                            end=max(df_pred_1.index.max(), df_val.index.max()),
                            freq='T')

    df_pred_1 = df_pred_1.reindex(time_index)
    df_pred_1 = df_pred_1.fillna(0)
    return df_pred_1

def interpolation_val():
    """对验证数据进行插值处理"""
    path = "data/Hengyuan1-301-values20180101-20181031.csv"
    def _val_data(path):
        hengyuan_data = pd.read_csv(path,
                        delimiter=",",
                        header=0,
                        parse_dates=['UTC Time'],
                        date_format='%d.%m.%y %H:%M')
        cum_quant_diff = hengyuan_data['Cum. Quant'].diff()
        utc_time_diff = (hengyuan_data['UTC Time'].diff().dt.total_seconds() / 3600).fillna(0)
        hengyuan_data['workingPower'] = 10 * (cum_quant_diff / utc_time_diff)
        hengyuan_data.loc[0, 'workingPower'] = 0
        df = hengyuan_data
        df['UTC Time'] = pd.to_datetime(df['UTC Time'])
        df.set_index('UTC Time', inplace=True)
        resampled_df = df.resample('T').interpolate(method='linear')
        return resampled_df[["workingPower"]]
    return _val_data(path)

def evaluate_predictions(dir_path):
    """评估预测信号的性能指标

    Args:
        dir_path (str): 包含预测信号文件的目录路径

    Returns:
        list: 包含评估结果的列表，每个元素是一个字典，包含文件名和对应的评估指标
    """
    results = []
    
    if not os.path.exists(dir_path):
        logging.warning(f"目录不存在 - {dir_path}")
        return results
        
    try:
        df_val = interpolation_val()
    except Exception as e:
        logging.error(f"加载验证数据失败: {str(e)}")
        return results

    valid_files = [f for f in os.listdir(dir_path) if not f.endswith('_error.csv')]

    if not valid_files:
        logging.warning(f"目录中没有有效的信号文件 - {dir_path}")
        return results

    for pred_path in valid_files:
        try:
            path = os.path.join(dir_path, pred_path)
            logging.info(f"处理文件 {path}")
            df_pred = interpolation_pred(path, df_val)

            vp_ratio = compute_infomation_ratio(df_pred, df_val)
            inclution = compute_inclution(df_pred, df_val)
            matching = compute_matching(df_pred, df_val)
            time_inclu = compute_time_inclution(df_pred, df_val)
            time_match = compute_time_matching(df_pred, df_val)
            power_errors = compute_power_error_by_state(df_pred, df_val)

            result = {
                'file_name': pred_path,
                'vp_ratio': vp_ratio,
                'inclution': inclution * 100,
                'matching': matching * 100,
                'time_inclution': time_inclu * 100,
                'time_matching': time_match * 100,
                'active_power_error': power_errors['active_power_error'],
                'inactive_power_error': power_errors['inactive_power_error'],
                'mse': compute_mse(df_pred, df_val),
                'mae': compute_mae(df_pred, df_val)
            }
            results.append(result)
            logging.info(result)
        
        except Exception as e:
            logging.error(f"处理文件 {pred_path} 时出错: {str(e)}")
            continue
    
    return results


def save_validation_events(validation_events, output_path):
    """保存验证集事件到CSV文件
    
    Args:
        validation_events (list): 验证集事件列表，每个事件为(start_time, end_time)元组
        output_path (str): 输出CSV文件路径
    """
    try:
        # 创建DataFrame
        events_data = []
        for i, (start_time, end_time) in enumerate(validation_events):
            # 转换时间格式为YYYY-MM-DD HH:mm:ss
            start_time = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_time = pd.to_datetime(end_time).strftime('%Y-%m-%d %H:%M:%S')
            events_data.append({
                'recordId': i + 1,
                'starttime': start_time,
                'endtime': end_time
            })
        
        df = pd.DataFrame(events_data)
        
        # 保存到CSV文件
        df.to_csv(output_path, index=False)
        logging.info(f'已保存验证集事件到文件: {output_path}')
        
    except Exception as e:
        logging.error(f'保存验证集事件时出错: {str(e)}')

def split_events_by_cluster(events_file_path, output_dir):
    """将_events.csv文件按聚类ID拆分成多个文件
    
    Args:
        events_file_path (str): _events.csv文件的路径
        
    Returns:
        dict: 包含每个聚类的事件数据，键为聚类ID，值为对应的事件DataFrame
    """
    try:
        # 读取_events.csv文件
        events_df = pd.read_csv(events_file_path)
        
        # 按clusterId分组
        cluster_groups = events_df.groupby('clusterId')
        
        # 创建输出目录
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个聚类创建单独的文件
        cluster_events = {}
        for cluster_id, group_df in cluster_groups:
            # 保存到字典中
            cluster_events[cluster_id] = group_df
            
            # 保存到文件
            output_file = os.path.join(output_dir, f'events_cluster_{cluster_id}.csv')
            group_df.to_csv(output_file, index=False)
            logging.info(f'已保存聚类 {cluster_id} 的事件到文件: {output_file}')
        
        return cluster_events
        
    except Exception as e:
        logging.error(f'拆分事件文件时出错: {str(e)}')
        return None

def evaluate_specific_output(output_prefix, img_dir, cluster_params=None):
    """评估特定输出目录的结果"""
    print(f"\n评估输出: {output_prefix}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_prefix)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.warning(f"无法创建输出目录: {str(e)}")

    # # 分析误差曲线
    # error_file = os.path.join(output_prefix, 'error', 'error.csv')
    # print("\n分析误差曲线...")
    # if os.path.exists(error_file):
    #     try:
    #         dw.plot_err(error_file)
    #         plt.show()
    #         plt.savefig(os.path.join(img_dir, 'error_curve.png'))
    #     except Exception as e:
    #         logging.error(f"绘制误差曲线失败: {str(e)}")

    # 评估分解结果
    print("\n开始评估分解结果...")
    logging.info(f"开始评估分解结果 - {output_prefix}")

    if cluster_params:
        logging.info(f"聚类参数 - {cluster_params}")

    try:
        data_path = os.path.join(output_prefix, "data")
        if not os.path.exists(data_path):
            logging.error(f"数据目录不存在 - {data_path}")
            return
            
        results = evaluate_predictions(data_path)
        
        if not results:
            logging.warning(f"未找到可评估的信号文件或评估结果为空 - {data_path}")
            return
            
        # 记录评估结果
        logging.info(f"找到 {len(results)} 个有效的评估结果")
        # score = [np.abs(np.log10(result["vp_ratio"]))/10+ result["active_power_error"]/10 + result["inactive_power_error"]/10 for result in results]
        score = [result["mse"]/10 for result in results]

        # score = [-result["matching"] for result in results]
        best_idx = np.argmin(score)
        best_result = results[best_idx]
        for result in results:
            logging.info(f"文件 {result['file_name']} 评估结果:")
            logging.info(f"  -vp_ratio: {result['vp_ratio']:.3f}")
            logging.info(f"  - inc: {result['inclution']:.3f}%")
            logging.info(f"  - mat: {result['matching']:.3f}%")
            logging.info(f"  - t_inc: {result['time_inclution']:.3f}%")
            logging.info(f"  - t_mat: {result['time_matching']:.3f}%")
            logging.info(f"  - act_mae: {result['active_power_error']:.3f}")
            logging.info(f"  - inact_mae:: {result['inactive_power_error']:.3f}")
            logging.info(f"  - mse: {result['mse']:.3f}")
            logging.info(f"  - mae: {result['mae']:.3f}")

        # 输出最佳匹配的信号文件
        logging.info(f"\n最佳匹配信号No.{best_idx+1}文件评估结果:")
        logging.info(f"  - vp_ratio: {best_result['vp_ratio']:.3f}")
        logging.info(f"  - inc: {best_result['inclution']:.3f}%")
        logging.info(f"  - mat: {best_result['matching']:.3f}%")
        logging.info(f"  - t_inc: {best_result['time_inclution']:.3f}%")
        logging.info(f"  - t_mat: {best_result['time_matching']:.3f}%")
        logging.info(f"  - act_mae: {best_result['active_power_error']:.3f}")
        logging.info(f"  - inact_mae:: {best_result['inactive_power_error']:.3f}")
        logging.info(f"  - mse: {best_result['mse']:.3f}")
        logging.info(f"  - mae: {best_result['mae']:.3f}")

            

    except Exception as e:
        logging.error(f"评估过程出错 - {output_prefix}: {str(e)}")
        print(f"评估过程出错: {str(e)}")
        return

    logging.info("评估完成\n" + "-"*50)

def visualize_well_operations(events_dir):
    """可视化各机井类型的运行情况
    
    Args:
        events_dir (str): 包含各聚类事件文件的目录路径
    """
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取所有聚类事件文件
    event_files = [f for f in os.listdir(events_dir) if f.startswith('events_cluster_') and f.endswith('.csv')]
    n_clusters = len(event_files)
    
    if n_clusters == 0:
        logging.warning(f"未找到聚类事件文件 - {events_dir}")
        return
    
    # 创建子图
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 3*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    
    # 设置颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 使用Set3色板，最多支持12种颜色
    
    # 获取所有事件的时间范围
    all_times = []
    for file in event_files:
        df = pd.read_csv(os.path.join(events_dir, file))
        all_times.extend(pd.to_datetime(df['starttime']))
        all_times.extend(pd.to_datetime(df['endtime']))
    
    time_range = [min(all_times), max(all_times)]
    
    # 处理每个聚类的事件
    for i, file in enumerate(sorted(event_files)):
        cluster_id = int(file.split('_')[2].split('.')[0])
        df = pd.read_csv(os.path.join(events_dir, file))
        
        # 转换时间列
        df['starttime'] = pd.to_datetime(df['starttime'])
        df['endtime'] = pd.to_datetime(df['endtime'])
        
        # 绘制事件区块
        ax = axes[i]
        for _, event in df.iterrows():
            # 计算运行机井数量
            wells = abs(event['aggNum']) if 'aggNum' in df.columns else 1
            # 绘制区块
            ax.fill_between([event['starttime'], event['endtime']], 
                          [wells, wells],
                          color=colors[i % len(colors)],
                          alpha=0.6)
        
        # 设置y轴范围和标签
        ax.set_ylim(0, max(df['aggNum'].max() if 'aggNum' in df.columns else 1, 1) + 0.5)
        ax.set_ylabel(f'Cluster {cluster_id}\nActive Wells', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 设置图表标题和布局
    fig.suptitle('Well Operations by Cluster', fontsize=14)
    plt.xlim(time_range)
    plt.tight_layout()
    
    return fig

def find_validation_set_events(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 检查必要的列是否存在
    required_columns = ['UTC Time', 'Cum. Quant']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f'CSV文件缺少必要的列: {required_columns}')
    
    # 确保时间列格式正确，支持混合格式和日期优先
    df['UTC Time'] = pd.to_datetime(df['UTC Time'], format='mixed', dayfirst=True)
    df = df.sort_values('UTC Time')  # 确保数据按时间顺序排列
    
    # 初始化事件列表
    validation_set_events = []
    
    # 初始化事件起始时间和结束时间
    start_time = None
    end_time = None
    prev_quant = None
    
    # 从前向后遍历数据
    for i in range(len(df)):
        current_time = df.iloc[i]['UTC Time']
        current_quant = df.iloc[i]['Cum. Quant']
        
        # 检查数据有效性
        if pd.isna(current_quant):
            continue
            
        # 第一个有效数据点
        if prev_quant is None:
            prev_quant = current_quant
            continue
            
        # 检测数据变化
        if current_quant != prev_quant:
            # 如果start_time为None，说明是新事件的开始
            if start_time is None:
                start_time = current_time
        else:
            # 如果start_time不为None，说明事件结束
            if start_time is not None:
                end_time = current_time
                # 将事件添加到列表中
                validation_set_events.append((
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S')
                ))
                # 重置start_time和end_time
                start_time = None
                end_time = None
        
        prev_quant = current_quant
    
    # 如果最后一个事件没有结束时间，则将其结束时间设置为最后一个时间点
    if start_time is not None:
        end_time = df.iloc[-1]['UTC Time']
        validation_set_events.append((
            start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    return validation_set_events

def validate_pump_events(validation_events, pump_data, threshold=0.1):
    """
    验证泵的启停事件是否与验证集事件匹配，并计算评估指标
    
    参数:
        validation_events: list of tuples, 验证集事件的(开始时间,结束时间)列表
        pump_data: DataFrame, 包含泵的实际运行数据，需要包含'UTC Time'和'workingPower'列
        threshold: float, 功率变化阈值，用于判断泵的启停状态
        
    返回:
        tuple: (validation_results, evaluation_metrics)
            validation_results: list of dicts, 每个字典包含验证事件的时间和对应的泵启停状态
            evaluation_metrics: dict, 包含各项评估指标
    """
    # 确保时间列格式正确
    pump_data['UTC Time'] = pd.to_datetime(pump_data['UTC Time'])
    pump_data.set_index('UTC Time', inplace=True)
    
    validation_results = []
    total_events = len(validation_events)
    detected_starts = 0
    detected_ends = 0
    total_time_diff_start = pd.Timedelta(0)
    total_time_diff_end = pd.Timedelta(0)
    total_power_similarity = 0
    
    for start_time, end_time in validation_events:
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        # 定义检测窗口
        start_window = start_time - pd.Timedelta(hours=1)
        end_window = end_time - pd.Timedelta(hours=1)
        
        # 获取窗口内的数据
        start_window_data = pump_data[start_window:start_time]
        end_window_data = pump_data[end_window:end_time]
        event_window_data = pump_data[start_time:end_time]
        
        # 检测启动波动
        power_changes_start = start_window_data['workingPower'].diff().abs()
        start_event_detected = any(power_changes_start > threshold)
        if start_event_detected:
            detected_starts += 1
            # 使用numpy的argmax找到最大功率变化点的位置
            max_change_idx = power_changes_start.values.argmax()
            start_event_time = start_window_data.index[max_change_idx]
            time_diff_start = abs(start_time - start_event_time)
            total_time_diff_start += time_diff_start
        
        # 检测关闭波动
        power_changes_end = end_window_data['workingPower'].diff().abs()
        end_event_detected = any(power_changes_end > threshold)
        if end_event_detected:
            detected_ends += 1
            # 使用numpy的argmax找到最大功率变化点的位置
            max_change_idx = power_changes_end.values.argmax()
            end_event_time = end_window_data.index[max_change_idx]
            time_diff_end = abs(end_time - end_event_time)
            total_time_diff_end += time_diff_end
        
        # 计算功率变化模式相似度
        if len(event_window_data) > 0:
            power_pattern = event_window_data['workingPower'].diff().abs()
            power_pattern_similarity = 1 - (power_pattern.std() / (power_pattern.mean() + 1e-6))
            total_power_similarity += power_pattern_similarity
        
        # 记录结果
        result = {
            'validation_start': start_time,
            'validation_end': end_time,
            'pump_start_detected': start_event_detected,
            'pump_end_detected': end_event_detected,
            'start_window': start_window,
            'end_window': end_window
        }
        if start_event_detected:
            result['start_event_time'] = start_event_time
            result['start_time_diff'] = time_diff_start
        if end_event_detected:
            result['end_event_time'] = end_event_time
            result['end_time_diff'] = time_diff_end
        
        validation_results.append(result)
    
    # 计算评估指标
    evaluation_metrics = {
        'start_detection_rate': detected_starts / total_events if total_events > 0 else 0,
        'end_detection_rate': detected_ends / total_events if total_events > 0 else 0,
        'avg_start_time_diff': (total_time_diff_start / detected_starts).total_seconds() / 60 if detected_starts > 0 else float('inf'),  # 转换为分钟
        'avg_end_time_diff': (total_time_diff_end / detected_ends).total_seconds() / 60 if detected_ends > 0 else float('inf'),  # 转换为分钟
        'avg_power_pattern_similarity': total_power_similarity / total_events if total_events > 0 else 0
    }
    
    return validation_results, evaluation_metrics

def interpret_validation_results(validation_results, evaluation_metrics):
    """
    解释验证结果并提供改进建议
    
    参数:
        validation_results: list of dicts, validate_pump_events返回的验证结果
        evaluation_metrics: dict, validate_pump_events返回的评估指标
    
    返回:
        dict: 包含解释和建议的字典
    """
    interpretation = {
        '总体评估': '',
        '具体分析': {},
        '改进建议': []
    }
    
    # 评估启停检测率
    start_rate = evaluation_metrics['start_detection_rate']
    end_rate = evaluation_metrics['end_detection_rate']
    avg_start_diff = evaluation_metrics['avg_start_time_diff']
    avg_end_diff = evaluation_metrics['avg_end_time_diff']
    power_similarity = evaluation_metrics['avg_power_pattern_similarity']
    
    # 总体评估
    if start_rate >= 0.8 and end_rate >= 0.8 and avg_start_diff <= 5 and avg_end_diff <= 5:
        interpretation['总体评估'] = '信号分解效果良好，能够准确捕捉大多数启停事件。'
    elif start_rate >= 0.6 and end_rate >= 0.6 and avg_start_diff <= 10 and avg_end_diff <= 10:
        interpretation['总体评估'] = '信号分解效果一般，可以捕捉到主要启停事件，但仍有改进空间。'
    else:
        interpretation['总体评估'] = '信号分解效果不理想，需要进行优化调整。'
    
    # 具体分析
    interpretation['具体分析'] = {
        '启动检测率': f'{start_rate*100:.1f}%，' + ('良好' if start_rate >= 0.8 else '需要改进'),
        '停止检测率': f'{end_rate*100:.1f}%，' + ('良好' if end_rate >= 0.8 else '需要改进'),
        '平均启动时间差': f'{avg_start_diff:.1f}分钟，' + ('良好' if avg_start_diff <= 5 else '需要改进'),
        '平均停止时间差': f'{avg_end_diff:.1f}分钟，' + ('良好' if avg_end_diff <= 5 else '需要改进'),
        '功率模式相似度': f'{power_similarity:.2f}，' + ('良好' if power_similarity >= 0.7 else '需要改进')
    }
    
    # 改进建议
    if start_rate < 0.8:
        interpretation['改进建议'].append('考虑调整启动检测的功率阈值，当前可能错过了一些小功率变化的启动事件')
    if end_rate < 0.8:
        interpretation['改进建议'].append('考虑调整停止检测的功率阈值，当前可能错过了一些渐变的停止事件')
    if avg_start_diff > 5 or avg_end_diff > 5:
        interpretation['改进建议'].append('检查时间窗口设置，可能需要调整检测窗口大小以提高时间精确度')
    if power_similarity < 0.7:
        interpretation['改进建议'].append('功率模式差异较大，建议检查信号分解算法是否正确提取了特征波形')
    
    return interpretation

def find_best_matching_pump(validation_events, pump_signals, threshold=0.1):
    """
    找出与验证集事件最匹配的机井类别
    
    参数:
        validation_events: list of tuples, 验证集事件的(开始时间,结束时间)列表
        pump_signals: dict, 键为机井类别名称，值为对应的泵信号数据DataFrame
        threshold: float, 功率变化阈值
    
    返回:
        tuple: (最匹配的机井类别, 匹配分数字典)
    """
    pump_scores = {}
    
    for pump_name, pump_data in pump_signals.items():
        # 对每个泵信号进行验证
        results, metrics = validate_pump_events(validation_events, pump_data, threshold)
        
        # 计算综合得分
        # 权重可以根据实际需求调整
        weights = {
            'start_detection_rate': 0.3,
            'end_detection_rate': 0.3,
            'time_accuracy': 0.2,  # 基于平均时间差
            'power_similarity': 0.2
        }
        
        # 将时间差转换为得分（时间差越小，得分越高）
        time_accuracy = 1.0 / (1.0 + (metrics['avg_start_time_diff'] + metrics['avg_end_time_diff']) / 2)
        
        # 计算加权得分
        score = (
            weights['start_detection_rate'] * metrics['start_detection_rate'] +
            weights['end_detection_rate'] * metrics['end_detection_rate'] +
            weights['time_accuracy'] * time_accuracy +
            weights['power_similarity'] * metrics['avg_power_pattern_similarity']
        )
        
        pump_scores[pump_name] = {
            'total_score': score,
            'details': {
                'start_detection_rate': metrics['start_detection_rate'],
                'end_detection_rate': metrics['end_detection_rate'],
                'time_accuracy': time_accuracy,
                'power_similarity': metrics['avg_power_pattern_similarity']
            }
        }
    
    # 找出得分最高的机井类别
    best_pump = max(pump_scores.items(), key=lambda x: x[1]['total_score'])
    
    return best_pump[0], pump_scores
