from datetime import datetime
import glob
from os.path import basename
import numpy as np
import pandas as pd

def check_dateparser(in_file):
    with open(in_file, "r") as f:
        f.readline()
        _ = f.readline().split(',')
        data_string = _[0]
    try:
        datetime.strptime(str(data_string), '%Y-%m-%d %H:%M:%S')
        dateparser_str = '%Y-%m-%d %H:%M:%S'
    except ValueError:
        try:
            datetime.strptime(str(data_string), '%m.%d.%Y %H:%M')
            dateparser_str = '%m.%d.%Y %H:%M'
        except ValueError:
            try:
                datetime.strptime(str(data_string), '%m.%d.%y %H:%M')
                dateparser_str = '%m.%d.%y %H:%M'
            except ValueError:
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