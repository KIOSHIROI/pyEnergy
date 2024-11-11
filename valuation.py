import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import pandas as pd
# 创建颜色数组cc
cc = np.array([
    [0.4000, 0.7608, 0.6471],
    [0.9882, 0.5529, 0.3843],
    [0.5529, 0.6275, 0.7961],
    [0.9059, 0.5412, 0.7647],
    [0.6510, 0.8471, 0.3294],
    [1.0000, 0.8510, 0.1843],
    [0.8980, 0.7686, 0.5804],
    [0.4510, 0.1020, 0.4549],
    [0.7804, 0.2706, 0.2745]
])

# 创建datetimeRange_start和datetimeRange_end列表
datetimeRange_start = [
    datetime.datetime(2018, 3, 1, 23, 0, 0),
    datetime.datetime(2018, 3, 29, 23, 0, 0),
    datetime.datetime(2018, 5, 29, 23, 0, 0),
    datetime.datetime(2018, 7, 6, 18, 0, 0),
    datetime.datetime(2018, 8, 8, 23, 0, 0),
    datetime.datetime(2018, 8, 24, 23, 0, 0),
    datetime.datetime(2018, 10, 16, 18, 0, 0)
]

datetimeRange_end = [
    datetime.datetime(2018, 3, 18, 23, 0, 0),
    datetime.datetime(2018, 4, 13, 23, 0, 0),
    datetime.datetime(2018, 6, 12, 23, 0, 0),
    datetime.datetime(2018, 7, 7, 23, 0, 0),
    datetime.datetime(2018, 8, 12, 23, 0, 0),
    datetime.datetime(2018, 8, 26, 23, 0, 0),
    datetime.datetime(2018, 10, 17, 23, 0, 0)
]

def get_val_data():
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
        resampled_df.reset_index(inplace=True)

        # 查看结果
        return resampled_df[['UTC Time', 'workingPower']].copy()
    return _val_data(path)

def plot_val(df):
    fig, axes = plt.subplots(4,2, figsize=(10,8))
    ax = axes.flatten()
    # 遍历日期时间范围
    for ii in range(len(datetimeRange_end)):
        a = ax[ii]

        # 绘制Hengyuan数据的面积图
        a.stackplot(df['UTC Time'], df['workingPower'], color='#646464', alpha=0.6)

        # 设置图形属性
        a.set_xlim([datetimeRange_start[ii], datetimeRange_end[ii]])
        a.set_ylim([0, 20])
        a.set_ylabel('Power [kW]')
        # a.grid(True)
        a.tick_params(axis="x", labelrotation=30)
        # 设置x轴的刻度为时间戳
        a.xaxis.set_major_locator(mdates.AutoDateLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d %Hh'))
    plt.tight_layout()
    plt.show()

def get_pred(df_val, df_pred):
    df_val.set_index('UTC Time')
    df_pred.set_index('UTC Time')
    target_index = df_val.index

    interpolated_pred = df_pred.reindex(target_index).interpolate(method='linear')
    return interpolated_pred

def get_rmse(df_pred, df_val):
    df_pred = get_pred(df_val, df_pred)
    mse = np.mean((df_pred.loc[:, 'workingPower'].values - df_val.loc[:, 'workingPower'].values) ** 2)
    rmse = np.sqrt(mse)
    return rmse
