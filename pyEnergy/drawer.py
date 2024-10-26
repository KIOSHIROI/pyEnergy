import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
# 假设 feature_standardized 是你的 DataFrame，并且已经包含了标准化后的数据
# 我们绘制前三个特征的三维散点图
def draw_3D_scatter(feature_standardized, feature_info):
    df = feature_standardized.iloc[:, :3] 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[feature_info[0]], df[feature_info[1]], df[feature_info[2]])

    ax.set_xlabel(feature_info[0])
    ax.set_ylabel(feature_info[1])
    ax.set_zlabel(feature_info[2])
 
    ax.set_title('3D Scatter Plot of the First Three Features')

    plt.show()

def draw_feature_hist(feature_standardized, feature_info):
    for feature in feature_info:
        if feature in feature_standardized.columns:
            if feature_standardized[feature].dtype != 'float64' and feature_standardized[feature].dtype != 'int64':
                feature_standardized[feature] = pd.to_numeric(feature_standardized[feature], errors='coerce')
                
            feature_standardized[feature].hist(bins=30, edgecolor='black')
            plt.title(feature)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column {feature} does not exist in DataFrame.")

def draw_corr(feature):
    #协方差分析
    corr = feature.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()


def draw_signal(data, param_str="curnt_B", ax=None):
    if ax == None:
        fig, ax1 = plt.subplots()
    signal = data[param_str]
    # 使用 fill_between 方法来创建面积图
    ax1.fill_between(data.index, 0, signal, color='#6be6d0', alpha=0.7, edgecolor='#717373', linewidth=1)
    ax1.set_facecolor("#f4f4f4")
    ax1.grid(True)
    date_ticks = pd.to_datetime(data.index)
    # 计算刻度间隔，每隔约10%的数据设置一个刻度
    tick_interval = max(1, len(date_ticks) // 10)  # 确保tick_interval为正数
    ax1.set_xticks(date_ticks[::tick_interval])
    ax1.set_xticklabels([dt.strftime('%H:%M') for dt in date_ticks[::tick_interval]], rotation=30)
    ax1.set_ylabel(param_str)
    ax1.set_ylim((0, signal.max()+0.1*(signal.max()-signal.min())))
    ax1.set_title(date_ticks[::tick_interval][0].strftime('%Y-%m-%d'), fontdict={"fontsize": 20})
    if ax == None:
        plt.show()


def draw_silhouette_scores(max_clusters, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def draw_parallel_coordinates(data, y_pred, colormap="Set1"):
    data = data.copy()
    data["Class"] = y_pred
    plt.figure(figsize=(10, 6))  # 设置图表大小
    ax = pd.plotting.parallel_coordinates(data, "Class", colormap=colormap)
    
    # 设置x轴标签旋转45度，避免重叠
    plt.xticks(rotation=45, fontsize=12)  # 可以调整字体大小
    plt.title('Parallel Coordinates Plot', fontsize=16)  # 设置标题和字体大小
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', which='major')
    
    # 设置图例位置和字体大小
    ax.legend(loc='upper right', fontsize=16)
    
    # 调整透明度
    for line in ax.get_lines():
        line.set_alpha(0.65)
        line.set_linewidth(0.65)
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()


def plot_continuous_lines(original_signals, reconstruct_signals, x_values=None):
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    if x_values is None:
        x_values = np.linspace(0, 1, len(reconstruct_signals))
    ax.set_ylim((0, original_signals.max()+0.1*(original_signals.max()-original_signals.min())))
    if len(x_values) <= len(original_signals):
        x_values = list(x_values)
        x_values = x_values +[x_values[-1]] * (len(original_signals) - len(x_values))
    else:
        x_values=list(x_values)[:len(original_signals)]
    ax.stackplot(x_values, original_signals, color=[213/255, 215/255, 215/255], alpha=0.7)

    last_value = None
    start_index = 0

    for i in range(len(reconstruct_signals)):
        if reconstruct_signals[i] != last_value:
            if last_value is not None:
                ax.plot([x_values[start_index], x_values[i-1]], [last_value, last_value], color='r', linestyle='dashed', linewidth=1.5)
            last_value = reconstruct_signals[i]
            start_index = i
    
    # 绘制最后一个不变段的水平线
    if last_value is not None:
        ax.plot([x_values[start_index], x_values[len(reconstruct_signals)-1]], [last_value, last_value], color='r', linestyle='dashed', linewidth=1.5, label="Reconstructed Signal")
        ax.set_xlim(x_values[0], x_values[-1])
    ax.legend()
    return fig

def plot_running_time(n_clusters, sols, cmap="viridis", n_color=15, color_reverse=False):
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, n_color))
    if color_reverse:
        colors = list(reversed(colors))
    fig,axes = plt.subplots(n_clusters, 1, figsize=(10, n_clusters))
    legends = [Rectangle((0.4, 0.4), 0.2, 0.2, facecolor=colors[i]) for i in range(n_color)]
    legend_labels = list(map(str, range(1, n_color+1)))
    fig.legend(legends, legend_labels)
    for i, ax in enumerate(axes):
        ax.set_ylabel("C{}".format(i + 1)) 
        ax.set_xlim(0, len(sols))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim((0,1))

    for i, sol in enumerate(sols):
        for j, ax in enumerate(axes):
            if sol[j] > 0:
                ax.fill_betweenx([0,1], i, i+1, color=colors[sol[j]-1])

    return fig

def draw_continue_line_and_running_time(signals, reconstruct_signals, n_clusters, sols, x_values=None, 
                                        cmap="summer", n_color=10, color_reverse=False, 
                                        save=None, plot=True):
    clfig = plot_continuous_lines(signals, reconstruct_signals, x_values=x_values)
    rtfig = plot_running_time(n_clusters, sols, cmap=cmap, n_color=n_color, color_reverse=color_reverse)
    if plot:
        plt.show()
    if save:
        clfig.savefig(save.split(".")[0]+"CL."+save.split(".")[-1])
        rtfig.savefig(save.split(".")[0]+"RT."+save.split(".")[-1])
