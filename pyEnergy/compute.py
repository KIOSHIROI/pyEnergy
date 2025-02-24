import pandas as pd
import os
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