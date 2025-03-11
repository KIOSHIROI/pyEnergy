import pandas as pd

def find_validation_set_events(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 初始化事件列表
    validation_set_events = []
    
    # 初始化事件起始时间和结束时间
    start_time = None
    end_time = None
    
    # 遍历数据（从最后一行开始，因为表格是倒序的）
    for i in range(len(df) - 1, 0, -1):
        # 如果当前值与上一个值不同，说明数据开始变化
        if df.iloc[i]['Cum. Quant'] != df.iloc[i-1]['Cum. Quant']:
            # 如果start_time为None，说明是新事件的开始
            if start_time is None:
                start_time = df.iloc[i-1]['UTC Time']  # 注意：这里用 i-1 作为起始时间
        # 如果当前值与上一个值相同，说明数据停止变化
        elif df.iloc[i]['Cum. Quant'] == df.iloc[i-1]['Cum. Quant']:
            # 如果start_time不为None，说明事件结束
            if start_time is not None:
                end_time = df.iloc[i]['UTC Time']
                # 将事件添加到列表中（确保起始时间早于结束时间）
                validation_set_events.append((start_time, end_time))  # 注意顺序
                # 重置start_time和end_time
                start_time = None
                end_time = None
    
    # 如果最后一个事件没有结束时间，则将其结束时间设置为第一个时间点
    if start_time is not None:
        end_time = df.iloc[0]['UTC Time']
        # 将事件添加到列表中（确保起始时间早于结束时间）
        validation_set_events.append((start_time, end_time))  # 注意顺序
    
    # 反转事件列表，使其按时间顺序排列
    validation_set_events = validation_set_events[::-1]
    
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

if __name__ == "__main__":

    file_path = "data\Hengyuan1-301-values20180101-20181031.csv"
    events = find_validation_set_events(file_path)
    # for event in events:
    #     print(event)
    pump_data = pd.read_csv("output\OLD_METHOD55\data\signal1of8.csv")
    results = validate_pump_events(events, pump_data)
    interpretation = interpret_validation_results(*results)
    
    # 按照字典结构打印解释结果
    print("\n总体评估:")
    print(interpretation['总体评估'])
    
    print("\n具体分析:")
    for key, value in interpretation['具体分析'].items():
        print(f"{key}: {value}")
    
    print("\n改进建议:")
    for suggestion in interpretation['改进建议']:
        print(f"- {suggestion}")
