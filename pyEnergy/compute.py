import pandas as pd

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

def compute_time_inclution(df_pred, df_val):
    diff = df_pred - df_val
    inclusion = 1 - (diff < 0).sum() / (df_val > 0).sum()
    return inclusion.iloc[0]

def compute_inclution(df_pred, df_val):
    diff = df_pred - df_val
    inclusion = 1 - df_val[diff < 0].sum() / df_val.sum()
    inclusion = inclusion.iloc[0]
    return inclusion

def compute_time_matching(df_pred, df_val):
    active = (df_val > 0).sum()
    matching = active / (df_pred > 0).sum()
    return matching.iloc[0]


def compute_matching(df_pred, df_val):
    diff = df_pred - df_val
    
    matching = (df_val.sum() - df_val[diff < 0].sum()) / df_pred.sum()
    return matching.iloc[0]


def interpolation_pred(path, df_val):   
    df_pred_1 = pd.read_csv(path)
    df_pred_1 = df_pred_1.drop(["Unnamed: 0"], axis=1)
    df_pred_1
    df_pred_1['UTC Time'] = pd.to_datetime(df_pred_1['UTC Time'])
    df_pred_1.set_index('UTC Time', inplace=True)
    df_pred_1 = df_pred_1.resample('T').interpolate(method='nearest')
    # df_pred_1.reset_index(inplace=True)
    time_index = pd.date_range(start=min(df_pred_1.index.min(), df_val.index.min()),
                            end=max(df_pred_1.index.max(), df_val.index.max()),
                            freq='T')

    df_pred_1 = df_pred_1.reindex(time_index)
    df_pred_1 = df_pred_1.fillna(0)
    return df_pred_1


def interpolation_val():
    path = "data/Hengyuan1-301-values20180101-20181031.csv"
    def _val_data(path):
        hengyuan_data = pd.read_csv(path,
                        delimiter=",",
                        header=0,
                        parse_dates=['UTC Time'],
                        date_format='%d.%m.%y %H:%M')
        cum_quant_diff = hengyuan_data['Cum. Quant'].diff()
    # 计算时间间隔（以小时为单位）
        utc_time_diff = (hengyuan_data['UTC Time'].diff().dt.total_seconds() / 3600).fillna(0)

    # 计算工作功率
        hengyuan_data['workingPower'] = 10 * (cum_quant_diff / utc_time_diff)

        # 将第一个工作功率值设置为0（因为第一个值是NaN）
        hengyuan_data.loc[0, 'workingPower'] = 0
        df = hengyuan_data
        df['UTC Time'] = pd.to_datetime(df['UTC Time'])
        df.set_index('UTC Time', inplace=True)
        resampled_df = df.resample('T').interpolate(method='linear')
        # resampled_df.reset_index(inplace=True)
        # 查看结果
        return resampled_df[["workingPower"]]
    return _val_data(path)