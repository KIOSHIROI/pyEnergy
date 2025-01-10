import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pyEnergy.CONST as CONST
import matplotlib.dates as mdates

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

def draw_signal_series(signal, ax=None):
    if ax == None:
        fig, ax1 = plt.subplots()
    ax1.fill_between(signal.index, 0, signal, color='#6be6d0', alpha=0.7, edgecolor='#717373', linewidth=1)
    ax1.set_facecolor("#f4f4f4")
    ax1.grid(True)
    date_ticks = pd.to_datetime(signal.index)
    # 计算刻度间隔，每隔约10%的数据设置一个刻度
    tick_interval = max(1, len(date_ticks) // 10)  # 确保tick_interval为正数
    ax1.set_xticks(date_ticks[::tick_interval])
    ax1.set_xticklabels([dt.strftime('%H:%M') for dt in date_ticks[::tick_interval]], rotation=30)
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
    fig, ax = plt.subplots(1,1, figsize=(10, 3))
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

def plot_div_signal(sols, paramPerCluster, color='blue', x_values=None):
    n_clusters = len(sols[0])
    sols = np.array(sols)
    x_lim = len(sols)
    sols = sols * np.array(paramPerCluster.reshape(1,n_clusters))
    sols = sols.T
    fig,axes = plt.subplots(n_clusters, 1, figsize=(10, n_clusters))
    for i, ax in enumerate(axes):
        top = sols[i].max()
        ax.set_ylabel("C{}".format(i + 1)) 
        # ax.set_xlim(0, x_lim)
        ax.set_ylim((0, top*1.1+0.5))

        for i, ax in enumerate(axes):
            ax.plot(x_values, sols[i], color=color)
    return fig

def draw_result(signals, reconstruct_signals,  sols, params_perCluster, x_values=None, save=None, plot=True):
    clfig = plot_continuous_lines(signals, reconstruct_signals, x_values=x_values)
    dvfig = plot_div_signal(sols, params_perCluster, x_values=x_values)
    stack = plot_stacked(sols, params_perCluster, x_values=x_values)
    if plot:
        plt.show()
    if save:
        clfig.savefig(save.split(".")[0]+"CL."+save.split(".")[-1])
        dvfig.savefig(save.split(".")[0]+"DV."+save.split(".")[-1])
        stack.savefig(save.split(".")[0]+"stack."+save.splot(".")[-1])
        

def plot_stacked(sols, paramPerCluster, x_values):
    n_clusters = len(sols[0])
    sols = np.array(sols)
    
    # 计算每个簇的数值
    sols = sols * np.array(paramPerCluster.reshape(1, n_clusters))
    sols = sols.T
    
    # 创建面积图
    x = x_values
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 绘制堆叠面积图
    ax.stackplot(x, sols, colors=plt.cm.Blues(np.linspace(0.3, 1, n_clusters)))
    
    # 设置标签和标题
    ax.set_xlabel("Time")
    ax.set_ylabel("Power")
    ax.set_title("Stacked Area Plot of Clusters")
    
    return fig

def plot_signal(df_list):
    datetimeRange_start = CONST.datetimeRange_start
    datetimeRange_end = CONST.datetimeRange_end
    fig, axes = plt.subplots(4,2, figsize=(10,8))
    ax = axes.flatten()
    for df in df_list:
        # 遍历日期时间范围
        for ii in range(len(datetimeRange_end)):
            a = ax[ii]

            # 绘制Hengyuan数据的面积图
            a.plot(df, alpha=0.6)

            # 设置图形属性
    for ii in range(len(datetimeRange_end)):
        a = ax[ii]
        a.set_xlim([datetimeRange_start[ii], datetimeRange_end[ii]])
        a.set_ylabel('Power [kW]')
        # a.grid(True)
        a.tick_params(axis="x", labelrotation=30)
        # 设置x轴的刻度为时间戳
        a.xaxis.set_major_locator(mdates.AutoDateLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Hh/%m/%d/'))
    plt.tight_layout()

def plot_err(path):
    err = pd.read_csv(path)
    plt.plot(err.loc[:, " mean_error"], label="".join(path.split("/")[-1].split('.')[:-1]))
    plt.legend()
    