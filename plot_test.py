import os
import pandas as pd
import matplotlib.pyplot as plt
from pyEnergy.drawer import plot_signal

# 配置路径和参数
signals_dir = 'output/CRT95_10/signals'
output_figure = 'signal_visualization.png'

# 加载所有CSV信号文件
df_list = []
for filename in os.listdir(signals_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(signals_dir, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df_list.append(df)

# 生成可视化图表
plt.figure(figsize=(12, 8))
plot_signal(df_list, 
           labels=[f'Cluster {i}' for i in range(len(df_list))],
           decomposed_labels=None)

# 保存并显示结果
plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches='tight')
plt.close()
print(f'可视化结果已保存至: {output_figure}')