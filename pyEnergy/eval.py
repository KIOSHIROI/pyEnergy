import os
import datetime
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pyEnergy.drawer as dw

def setup_logging():
    """设置日志记录"""
    log_dir = os.path.join('output', 'log')
    os.makedirs(log_dir, exist_ok=True)
    _ = os.path.join(log_dir, f'{datetime.datetime.now().strftime("%Y%m%d")}')
    os.makedirs(_, exist_ok=True)
    log_file = os.path.join(_, datetime.datetime.now().strftime("%H%M%S") + '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def compute_time_inclution(df_pred, df_val):
    """计算时间包含度"""
    diff = df_pred - df_val
    inclusion = 1 - (diff < 0).sum() / (df_val > 0).sum()
    return inclusion.iloc[0]

def compute_inclution(df_pred, df_val):
    """计算功率包含度"""
    diff = df_pred - df_val
    inclusion = 1 - df_val[diff < 0].sum() / df_val.sum()
    inclusion = inclusion.iloc[0]
    return inclusion

def compute_time_matching(df_pred, df_val):
    """计算时间匹配度：关注时间点的重叠匹配情况"""
    pred_active = df_pred > 0
    val_active = df_val > 0
    overlap = (pred_active & val_active).sum()
    total_active = (pred_active | val_active).sum()
    matching = overlap.iloc[0] / total_active.iloc[0] if total_active.iloc[0] > 0 else 0
    return matching

def compute_matching(df_pred, df_val):
    """计算功率匹配度：关注功率值的匹配情况"""
    diff = df_pred - df_val
    matching = (df_val.sum() - df_val[diff < 0].sum()) / df_pred.sum()
    return matching.iloc[0]

def compute_power_error_by_state(df_pred, df_val):
    """计算在有信号和无信号时段的功率误差

    Args:
        df_pred: 预测的功率数据
        df_val: 验证集的功率数据

    Returns:
        dict: 包含有信号和无信号时段的平均功率误差
    """
    power_error = abs(df_pred - df_val)
    signal_mask = df_val > 0
    active_error = power_error[signal_mask].mean().iloc[0] if signal_mask.any().iloc[0] else 0
    inactive_error = power_error[~signal_mask].mean().iloc[0] if (~signal_mask).any().iloc[0] else 0
    
    return {
        'active_power_error': active_error,
        'inactive_power_error': inactive_error
    }

def compute_mse(df_pred, df_val):
    """计算均方误差（MSE）"""
    diff = df_pred - df_val
    mse = np.mean(diff**2)
    return mse

def compute_infomation_ratio(df_pred, df_val):
    n_pred = df_pred.sum().iloc[0]
    n_val = df_val.sum().iloc[0]
    return n_val / n_pred

def compute_mae(df_pred, df_val):
    """计算平均绝对误差（MAE）"""
    diff = abs(df_pred - df_val)
    mae = np.mean(diff)
    return mae


def interpolation_pred(path, df_val):
    """对预测数据进行插值处理"""
    df_pred_1 = pd.read_csv(path)
    df_pred_1 = df_pred_1.drop(["Unnamed: 0"], axis=1)
    df_pred_1['UTC Time'] = pd.to_datetime(df_pred_1['UTC Time'])
    df_pred_1.set_index('UTC Time', inplace=True)
    df_pred_1 = df_pred_1.resample('T').interpolate(method='nearest')
    time_index = pd.date_range(start=min(df_pred_1.index.min(), df_val.index.min()),
                            end=max(df_pred_1.index.max(), df_val.index.max()),
                            freq='T')

    df_pred_1 = df_pred_1.reindex(time_index)
    df_pred_1 = df_pred_1.fillna(0)
    return df_pred_1

def interpolation_val():
    """对验证数据进行插值处理"""
    path = "data/Hengyuan1-301-values20180101-20181031.csv"
    def _val_data(path):
        hengyuan_data = pd.read_csv(path,
                        delimiter=",",
                        header=0,
                        parse_dates=['UTC Time'],
                        date_format='%d.%m.%y %H:%M')
        cum_quant_diff = hengyuan_data['Cum. Quant'].diff()
        utc_time_diff = (hengyuan_data['UTC Time'].diff().dt.total_seconds() / 3600).fillna(0)
        hengyuan_data['workingPower'] = 10 * (cum_quant_diff / utc_time_diff)
        hengyuan_data.loc[0, 'workingPower'] = 0
        df = hengyuan_data
        df['UTC Time'] = pd.to_datetime(df['UTC Time'])
        df.set_index('UTC Time', inplace=True)
        resampled_df = df.resample('T').interpolate(method='linear')
        return resampled_df[["workingPower"]]
    return _val_data(path)

def evaluate_predictions(dir_path):
    """评估预测信号的性能指标

    Args:
        dir_path (str): 包含预测信号文件的目录路径

    Returns:
        list: 包含评估结果的列表，每个元素是一个字典，包含文件名和对应的评估指标
    """
    results = []
    
    if not os.path.exists(dir_path):
        logging.warning(f"目录不存在 - {dir_path}")
        return results
        
    try:
        df_val = interpolation_val()
    except Exception as e:
        logging.error(f"加载验证数据失败: {str(e)}")
        return results

    valid_files = [f for f in os.listdir(dir_path) if not f.endswith('_error.csv')]

    if not valid_files:
        logging.warning(f"目录中没有有效的信号文件 - {dir_path}")
        return results

    for pred_path in valid_files:
        try:
            path = os.path.join(dir_path, pred_path)
            logging.info(f"处理文件 {path}")
            df_pred = interpolation_pred(path, df_val)

            vp_ratio = compute_infomation_ratio(df_pred, df_val)
            inclution = compute_inclution(df_pred, df_val)
            matching = compute_matching(df_pred, df_val)
            time_inclu = compute_time_inclution(df_pred, df_val)
            time_match = compute_time_matching(df_pred, df_val)
            power_errors = compute_power_error_by_state(df_pred, df_val)

            result = {
                'file_name': pred_path,
                'vp_ratio': vp_ratio,
                'inclution': inclution * 100,
                'matching': matching * 100,
                'time_inclution': time_inclu * 100,
                'time_matching': time_match * 100,
                'active_power_error': power_errors['active_power_error'],
                'inactive_power_error': power_errors['inactive_power_error'],
                'mse': compute_mse(df_pred, df_val),
                'mae': compute_mae(df_pred, df_val)
            }
            results.append(result)
            logging.info(result)
        
        except Exception as e:
            logging.error(f"处理文件 {pred_path} 时出错: {str(e)}")
            continue
    
    return results


def evaluate_specific_output(output_prefix, img_dir, cluster_params=None):
    """评估特定输出目录的结果"""
    print(f"\n评估输出: {output_prefix}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_prefix)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.warning(f"无法创建输出目录: {str(e)}")

    # # 分析误差曲线
    # error_file = os.path.join(output_prefix, 'error', 'error.csv')
    # print("\n分析误差曲线...")
    # if os.path.exists(error_file):
    #     try:
    #         dw.plot_err(error_file)
    #         plt.show()
    #         plt.savefig(os.path.join(img_dir, 'error_curve.png'))
    #     except Exception as e:
    #         logging.error(f"绘制误差曲线失败: {str(e)}")

    # 评估分解结果
    print("\n开始评估分解结果...")
    logging.info(f"开始评估分解结果 - {output_prefix}")

    if cluster_params:
        logging.info(f"聚类参数 - {cluster_params}")

    try:
        data_path = os.path.join(output_prefix, "data")
        if not os.path.exists(data_path):
            logging.error(f"数据目录不存在 - {data_path}")
            return
            
        results = evaluate_predictions(data_path)
        
        if not results:
            logging.warning(f"未找到可评估的信号文件或评估结果为空 - {data_path}")
            return
            
        # 记录评估结果
        logging.info(f"找到 {len(results)} 个有效的评估结果")
        # score = [np.abs(np.log10(result["vp_ratio"]))/10+ result["active_power_error"]/10 + result["inactive_power_error"]/10 for result in results]
        score = [result["mse"]/10 for result in results]

        # score = [-result["matching"] for result in results]
        best_idx = np.argmin(score)
        best_result = results[best_idx]
        for result in results:
            logging.info(f"文件 {result['file_name']} 评估结果:")
            logging.info(f"  -vp_ratio: {result['vp_ratio']:.3f}")
            logging.info(f"  - inc: {result['inclution']:.3f}%")
            logging.info(f"  - mat: {result['matching']:.3f}%")
            logging.info(f"  - t_inc: {result['time_inclution']:.3f}%")
            logging.info(f"  - t_mat: {result['time_matching']:.3f}%")
            logging.info(f"  - act_mae: {result['active_power_error']:.3f}")
            logging.info(f"  - inact_mae:: {result['inactive_power_error']:.3f}")
            logging.info(f"  - mse: {result['mse']:.3f}")
            logging.info(f"  - mae: {result['mae']:.3f}")

        # 输出最佳匹配的信号文件
        logging.info(f"\n最佳匹配信号No.{best_idx+1}文件评估结果:")
        logging.info(f"  - vp_ratio: {best_result['vp_ratio']:.3f}")
        logging.info(f"  - inc: {best_result['inclution']:.3f}%")
        logging.info(f"  - mat: {best_result['matching']:.3f}%")
        logging.info(f"  - t_inc: {best_result['time_inclution']:.3f}%")
        logging.info(f"  - t_mat: {best_result['time_matching']:.3f}%")
        logging.info(f"  - act_mae: {best_result['active_power_error']:.3f}")
        logging.info(f"  - inact_mae:: {best_result['inactive_power_error']:.3f}")
        logging.info(f"  - mse: {best_result['mse']:.3f}")
        logging.info(f"  - mae: {best_result['mae']:.3f}")

            

    except Exception as e:
        logging.error(f"评估过程出错 - {output_prefix}: {str(e)}")
        print(f"评估过程出错: {str(e)}")
        return

    logging.info("评估完成\n" + "-"*50)