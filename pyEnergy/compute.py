import numpy as np
import pandas as pd
def calc_dominant_bin(inArray, bin_interval):
    '''
    计算主导区间中点
    '''
    hist, bins = np.histogram(inArray, bins=bin_interval)
    idx = hist.argmax()
    dominant_bin = (bins[idx] + bins[idx + 1]) / 2

    return dominant_bin

def z_score(data):
    df = pd.DataFrame(data)
    mean = df.mean()
    std = df.std()

    # 进行标准化
    df_standardized = (df - mean) / std

    return df_standardized

def min_max(data):
    df = pd.DataFrame(data)
    max = df.max()
    min = df.min()

    # 进行标准化
    df_normalized = (df - min) / (max - min)

    return df_normalized