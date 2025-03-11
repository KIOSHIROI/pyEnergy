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

# if __name__ == "__main__":
#     file_path = "data\Hengyuan1-301-values20180101-20181031.csv"
#     events = find_validation_set_events(file_path)
#     for event in events:
#         print(event)