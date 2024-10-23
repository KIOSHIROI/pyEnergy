import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from os.path import basename
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns


def calc_dominant_bin(inArray, bin_interval):
    """
    计算给定数组的主导区间（出现频率最高的区间）。

    参数:
    inArray (np.array): 输入的数值数据数组。
    bin_interval (int或数组形式): 区间的数量或各区间的边界。

    返回:
    float: 主导区间的中间值。
    """

    # 计算直方图
    hist, bins = np.histogram(inArray, bins=bin_interval)

    # 查找计数最高的区间的索引
    idx = hist.argmax()

    # 计算主导区间的中点
    dominant_bin = (bins[idx] + bins[idx + 1]) / 2

    return dominant_bin


def check_dateparser(in_file):
    # 读取文件的第一行数据
    # 尝试解析日期格式
    with open(in_file, "r") as f:
        f.readline()
        _ = f.readline().split(',')
        data_string = _[0]

    try:
        datetime.strptime(str(data_string), '%Y-%m-%d %H:%M:%S')
        dateparser_str = '%Y-%m-%d %H:%M:%S'
    except ValueError:
        # 如果第一个格式不匹配，尝试第二个格式
        try:
            datetime.strptime(str(data_string), '%d.%m.%Y %H:%M')
            dateparser_str = '%d.%m.%Y %H:%M'
        except ValueError:
            # 如果都不匹配，可能需要处理这种情况或抛出异常
            raise ValueError("无法识别的日期格式")

    return dateparser_str


def check_nr_transformer(path_to_data):
    """
    确定不同变压器的数量，并将它们对应的 CSV 文件按标识符分组。

    参数:
    path_to_data (str): 包含数据文件的目录的路径。

    返回:
    dict: 一个字典，其中每个键是一个变压器标识符，值是文件路径列表。
    """

    # 使用 glob 查找指定目录中的所有 CSV 文件
    data_files = glob(path_to_data + "/*.csv")

    transformer = []
    # 从每个文件名中提取变压器标识符
    for each_file in data_files:
        # 假定文件名格式为 "<transformerID>-<additional_info>.csv"
        tmp = basename(each_file).split('-')
        transformer.append(tmp[0] + '-' + tmp[1])

    # 获取唯一的变压器标识符
    transformer_unique = np.unique(transformer)

    transformer_dict = {}
    # 按变压器分组文件
    for each_transformer in transformer_unique:
        # 调整 glob 模式以匹配每个变压器的文件
        files = glob(path_to_data + "/" + each_transformer + "*.csv")
        transformer_dict[each_transformer] = files

    return transformer_dict


def import_transformer_data(data_files):
    selected_cols = range(0, 26)
    param_names = ['T',
                   'volt_A', 'volt_B', 'volt_C', 'volt_AB', 'volt_CA', 'volt_BC',
                   'curnt_A', 'curnt_B', 'curnt_C',
                   'realP_A', 'realP_B', 'realP_C', 'realP_tot',
                   'reacP_A', 'reacP_B', 'reacP_C', 'reacP_tot',
                   'aprntP_A', 'aprntP_B', 'aprntP_C', 'aprntP_tot',
                   'factor_A', 'factor_B', 'factor_C', 'factor_tot']

    date_parser_str = check_dateparser(data_files)
    df = pd.read_csv(data_files,
                     delimiter=",",
                     header=0,
                     names=param_names,
                     index_col='T',
                     parse_dates=['T'],
                     date_format=date_parser_str,  # 使用date_format代替date_parser
                     usecols=range(0, 26))

    idx_duplicate = df.index.duplicated(keep='first')
    df = df[~idx_duplicate].sort_index()

    df.loc[:, 'realP_A':'aprntP_tot'] /= 1000  # 单位转换：瓦特到千瓦
    df.loc[:, 'factor_A':'factor_tot'] = abs(df.loc[:, 'factor_A':'factor_tot'])  # 功率因数转为正值

    return df


def find_all_events(df, thre_val, thre_time):
    """
    根据实际功率使用情况获取事件。

    参数:
    df (pd.DataFrame): 包含数据的 DataFrame。
    thre_val (float): 电流阈值，低于此值的数据将被视为无效。
    thre_time (float): 事件持续时间的最小阈值（分钟）。

    返回:
    tuple: 包含有效事件持续时间和所有有效事件的列表。
    """
# TODO: Why 'curnt_B'
    param_str = 'curnt_B'
    signals = df[param_str].copy()
    idx_invalid = signals < thre_val
    signals[idx_invalid] = 0

    idx_valid = signals > 0
    idx_valid = np.insert(idx_valid.values.astype(int), 0, 0)

    idx_start_event = np.where(np.diff(idx_valid) == 1)[0]
    idx_end_event = np.where(np.diff(idx_valid) == -1)[0] - 1

    # 确保每个开始事件都有对应的结束事件
    if len(idx_start_event) > len(idx_end_event):
        idx_start_event = idx_start_event[0:len(idx_end_event)]  # 假设最后一个事件持续到数据末尾

    event_durations = (df.index[idx_end_event] - df.index[idx_start_event]).total_seconds() / 60

    # 根据持续时间筛选有效事件
    valid_events_mask = event_durations > thre_time
    valid_event_durations = event_durations[valid_events_mask]
    valid_start_idx = idx_start_event[valid_events_mask]
    valid_end_idx = idx_end_event[valid_events_mask]

    events_all = [df.iloc[start:end + 1] for start, end in zip(valid_start_idx, valid_end_idx)]

    return valid_event_durations, events_all


def find_monotype_events(events_all, thre_val):
    """
    将所有有效事件分为“单一类型事件”和其他事件。

    参数:
    events_all (list): 包含所有事件的列表，每个事件是一个包含所有参数的 pandas DataFrame。
    thre_val (float): 用于判断事件是否为单一类型的方差阈值。

    返回:
    tuple: 包含三个元素的元组。
           第一个元素是单一类型事件的列表；
           第二个元素是一个布尔数组，表示每个事件是否为单一类型；
           第三个元素是其他事件的列表。
    """

    param_str = 'curnt_B'

    monotype_events = []
    other_events = []
    idx_monotype = np.zeros(len(events_all), dtype=bool)

    for i, event_ in enumerate(events_all):
        signal = event_[param_str].values

        # 使用方差来判断事件类型
        if np.var(signal) < thre_val:
            monotype_events.append(event_)
            idx_monotype[i] = True
        else:
            other_events.append(event_)

    return monotype_events, idx_monotype, other_events


def estimate_total_power(event):
    """
    估计泵的总功率，包括有功功率和无功功率。

    参数:
    event (pd.DataFrame): 包含事件数据的 DataFrame，应包括电压、电流和功率因数的值。

    返回:
    pd.DataFrame: 更新后包含有功功率和无功功率估计的事件 DataFrame。
    """

    # 电压 [V]
    v_A = event['volt_A'].values
    v_B = event['volt_B'].values
    v_C = event['volt_C'].values

    # 电流 [A]
    c_B = event['curnt_B'].values
    c_C = event['curnt_C'].values

    # 功率因数 [-]
    pf_A = event['factor_A'].values
    pf_B = event['factor_B'].values
    pf_C = event['factor_C'].values

    # 估算泵的总电压、电流和功率因数
    ave_pf = np.mean([pf_A, pf_B, pf_C], axis=0)
    ave_curnt = np.mean([c_B, c_C], axis=0)
    ave_volt = np.mean([v_A, v_B, v_C], axis=0)

    # 计算有功功率和无功功率（以千瓦为单位）
    realP_proxy = 3 * ave_pf * ave_curnt * ave_volt / 1000
    reactP_proxy = 3 * np.sqrt(1 - np.square(ave_pf)) * ave_curnt * ave_volt / 1000

    # 将功率数据添加到事件 DataFrame
    event = event.assign(realP_proxy=realP_proxy, reactP_proxy=reactP_proxy)

    return event


def compute_features_mono(monotype_events):
    # 定义要计算的特征名
    feature_info = [
        'std. real power(ss)', 'ave. real power(ss)', 'max. real power(tr)',
        'std. reactive power(ss)', 'ave. reactive power(ss)', 'max. reactive power(tr)',
        'std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)'
    ]

    tr_steps = 5  # 暂态阶段的时间步数定义
    num_events = len(monotype_events)  # 事件总数

    # 创建一个布尔数组，用于标记每个事件是否被用于计算特征
    idx_used_events = np.ones(num_events, dtype=bool)
    # 创建一个空的DataFrame，用于存储所有计算出的特征
    feature_list = pd.DataFrame(index=range(num_events), columns=feature_info)

    # 遍历每个事件，计算特征
    for ctr, event_ in enumerate(monotype_events):
        # 提取暂态阶段的有功功率、无功功率和B相电流
        tr_realP = event_['realP_proxy'][0:tr_steps]
        tr_reactP = event_['reactP_proxy'][0:tr_steps]
        tr_curntB = event_['curnt_B'][0:tr_steps]

        # 提取稳态阶段的有功功率、无功功率和B相电流
        ss_realP = event_['realP_proxy'][tr_steps:]
        ss_reactP = event_['reactP_proxy'][tr_steps:]
        ss_curntB = event_['curnt_B'][tr_steps:]

        # 如果稳态阶段的样本数量少于10，不使用此事件计算特征
        if len(ss_realP) < 10:
            idx_used_events[ctr] = False
            continue

        feature_list.loc[ctr, 'std. real power(ss)'] = np.std(ss_realP)
        feature_list.loc[ctr, 'ave. real power(ss)'] = np.mean(ss_realP)
        feature_list.loc[ctr, 'max. real power(tr)'] = np.max(tr_realP)

        feature_list.loc[ctr, 'std. reactive power(ss)'] = np.std(ss_reactP)
        feature_list.loc[ctr, 'ave. reactive power(ss)'] = np.mean(ss_reactP)
        feature_list.loc[ctr, 'max. reactive power(tr)'] = np.max(tr_reactP)

        feature_list.loc[ctr, 'std. phase B current(ss)'] = np.std(ss_curntB)
        feature_list.loc[ctr, 'ave. phase B current(ss)'] = np.mean(ss_curntB)
        feature_list.loc[ctr, 'max. phase B current(tr)'] = np.max(tr_curntB)

    feature_list = feature_list[idx_used_events]

    return feature_list, idx_used_events


def compute_cluster_score(feature_val, grps, w):
    """
    计算基于轮廓系数和其标准差的聚类性能评分。

    参数:
    feature_val (array-like): 特征数据，每一行是一个数据点，每一列是一个特征。
    grps (array-like): 每个数据点的聚类标签。
    w (list or array): 两个权重系数，分别对应轮廓系数的平均值和标准差的权重。

    返回:
    float: 计算出的性能值。
    """

    # 计算每个点的轮廓系数
    scores = silhouette_samples(feature_val, grps)

    # 初始化用于存储每个聚类轮廓系数标准差的数组
    std_tmp = np.zeros(len(np.unique(grps)))

    # 计算每个聚类的轮廓系数标准差
    for i in np.unique(grps):
        idx = np.where(grps == i)
        std_tmp[i] = np.std(scores[idx])

    # 计算轮廓系数的平均值和标准差的平均值
    S_coeff = np.mean(scores)
    score_std = -np.mean(std_tmp)  # 负号表示我们希望最大化总得分

    # 计算最终的性能评分
    perf_val = w[0] * S_coeff + w[1] * score_std

    return perf_val


def kmeans_elbow(feature_list, normalize=False, max_iter=None, repeats=5):
    # 数据标准化处理
    if normalize:
        max_val = np.max(feature_list, axis=0)
        min_val = np.min(feature_list, axis=0)
        feature_list = (feature_list - min_val) / (max_val - min_val)

    # 默认最大迭代次数
    if max_iter is None:
        max_iter = np.ceil(np.sqrt(len(feature_list))).astype(int)
    print("max_iteration: ", max_iter)
    scores_final = np.zeros(max_iter)
    scores_final[0] = -np.inf  # 初始化第一个得分为负无穷

    grp_opt = [np.nan]  # 初始化最优群组列表

    # 遍历每一个可能的聚类数目
    for ctr in range(2, max_iter + 1):
        grp_opt_tmp = KMeans(n_clusters=ctr, n_init='auto').fit_predict(feature_list)
        perf_val_opt = compute_cluster_score(feature_list, grp_opt_tmp, [0.5, 0.5])

        # 重复运算以寻找更优解
        for _ in range(repeats):
            grp_tmp = KMeans(n_clusters=ctr, n_init='auto').fit_predict(feature_list)
            perf_val_tmp = compute_cluster_score(feature_list, grp_tmp, [0.5, 0.5])

            if perf_val_tmp > perf_val_opt:
                grp_opt_tmp = grp_tmp
                perf_val_opt = perf_val_tmp

        scores_final[ctr - 1] = perf_val_opt
        grp_opt.append(grp_opt_tmp)

    # 找出性能值最高的聚类数
    nrCluster_opt = np.argmax(scores_final) + 1
    grp_opt_final = grp_opt[nrCluster_opt - 1]

    # 计算最优群组的最终得分
    scores_opt = compute_cluster_score(feature_list, grp_opt_final, [0.5, 0.5])

    return grp_opt_final, scores_opt


def create_figure_and_plot(data, field):
    fig, ax = plt.subplots()
    # 使用 fill_between 方法来创建面积图
    ax.fill_between(data.index, 0, data[field], color='#6be6d0', alpha=0.7, edgecolor='#717373', linewidth=1)
    return ax


def customize_plot_appearance(ax):
    ax.set_facecolor("#f4f4f4")  # 设置绘图区域的背景色
    ax.grid(True)


def setup_axes(ax, data, ylabel_text):
    date_ticks = pd.to_datetime(data.index)
    # 计算刻度间隔，每隔约10%的数据设置一个刻度
    tick_interval = max(1, len(date_ticks) // 10)  # 确保tick_interval为正数
    ax.set_xticks(date_ticks[::tick_interval])
    ax.set_xticklabels([dt.strftime('%H:%M') for dt in date_ticks[::tick_interval]], rotation=30)
    ax.set_ylabel(ylabel_text)
    ax.set_title(date_ticks[::tick_interval][0].strftime('%Y-%m-%d'))


def plot_ts_area(data, param_str):
    ax = create_figure_and_plot(data, param_str)
    customize_plot_appearance(ax)
    setup_axes(ax, data, param_str)
    plt.tight_layout()
    plt.show()


def perform_pca(data, n_components=2):
    """执行 PCA 分析并返回分析结果。"""
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)
    coeff = pca.components_.T
    score = pca_results
    latent = pca.explained_variance_
    tsquared = np.sum(pca_results ** 2, axis=1)
    explained = pca.explained_variance_ratio_
    return coeff, score, latent, tsquared, explained


def plot_biplot(coeff, score, feature_names):
    """绘制 PCA 双标图。"""
    plt.figure(figsize=(8, 6))
    plt.scatter(score[:, 0], score[:, 1], c='r', s=2, label='Scores')
    for i, vec in enumerate(coeff):
        plt.arrow(0, 0, vec[0] * max(score[:, 0]), vec[1] * max(score[:, 1]), color='b', width=0.001, head_width=0.002)
        plt.text(vec[0] * max(score[:, 0]) * 1.1, vec[1] * max(score[:, 1]) * 1.1, feature_names[i], color='black',
                 fontsize=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()


def calculate_correlation(data):
    """计算并返回数据的相关系数矩阵。"""
    return data.corr()


def plot_correlation_matrix(cor_matrix, feature_names):
    """绘制相关系数矩阵的热图，并优化显示设置。"""
    plt.figure(figsize=(8, 8))
    sns.heatmap(cor_matrix, annot=False, cmap='RdBu', center=0,
                linewidths=10, linecolor='white',  # 增强网格线显示
                vmin=-1, vmax=1,  # 设置颜色映射的范围
                square=True)  # 保持每个单元格的方形比例
    plt.xticks(ticks=np.arange(0.5, len(feature_names)), labels=feature_names, rotation=90)
    plt.yticks(ticks=np.arange(0.5, len(feature_names)), labels=feature_names)
    plt.title('Correlation Matrix')
    plt.colorbar()  # 显示色条
    plt.show()


def analyze_data(data):
    """执行数据的 PCA 分析和相关性矩阵绘图。"""
    coeff, score, latent, tsquared, explained = perform_pca(data)
    print("Explained variance by component:", explained)
    plot_biplot(coeff, score, data.columns)
    cor_matrix = calculate_correlation(data)
    plot_correlation_matrix(cor_matrix, data.columns)


def normalize_dataframe(df, feature_range=(0, 1)):
    """
    对输入的 DataFrame 的每一列进行归一化处理，将数值缩放到给定的范围内。

    参数:
    df (pandas.DataFrame): 需要被归一化的 DataFrame。
    feature_range (tuple, optional): 归一化的目标范围，默认为 (0, 1)。

    返回:
    pandas.DataFrame: 归一化后的 DataFrame。
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    # 只归一化数值型列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
