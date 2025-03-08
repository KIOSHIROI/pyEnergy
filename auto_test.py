import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import pyEnergy.CONST as CONST
from pyEnergy.composition.composition import Composer, auto_compose
from pyEnergy.fool import Fool
from pyEnergy.clusters.two_stage import TwoStageCluster
from pyEnergy.clusters.kmeans import Kmeans
from pyEnergy.clusters.HAC import HAC
from pyEnergy.clusters.mixture import Gaussian
from pyEnergy.eval import evaluate_specific_output, setup_logging

def auto_test(fool, model_class, is_two_stage=False, weights=None, stage2_method=None):
    """自动测试不同聚类方法和两阶段聚类
    
    参数:
        fool: Fool对象，包含特征数据
        model_class: 聚类模型类（HAC、Gaussian等）
        is_two_stage: 是否使用两阶段聚类
        weights: 权重列表，用于两阶段聚类
        stage2_method: 第二阶段聚类方法类（可选，仅在is_two_stage=True时有效）
    """
    # 聚类参数配置
    cluster_params = {
        'min_samples': 1,
        'repeats': 10,
        'plot': False,
        'weights': weights if weights else [0.5, 0.5]
    }
    
    # 创建模型
    if is_two_stage:
        stage1_method = model_class.__name__.lower()
        stage2_method_name = stage2_method.__name__.lower() if stage2_method else stage1_method
        model = TwoStageCluster(fool, stage1_method=stage1_method, stage2_method=stage2_method_name)
        model_name = f"{model_class.__name__}_{stage2_method.__name__ if stage2_method else model_class.__name__}_two_stage_w{''.join(map(str, cluster_params['weights']))}"
    else:
        model = model_class(fool)
        model_name = f"OLD_METHOD_f{model_class.__name__}"
    
    # 执行聚类分析
    y_pred, score, n_clusters = model.fit(**cluster_params)
    print(f"聚类完成: 最佳簇数={n_clusters}")
    
    # 负荷分解配置
    composer = Composer(model.fool, y_pred, threshold=1)
    composer.set_param('curnt_B', fit=False, threshold=0.5)
    composer.set_reducer('my2')
    
    # 执行负荷分解

    output_prefix = f"output/{model_name}/data/{model_name}"
    print(f"开始负荷分解，输出文件前缀: {output_prefix}")
    auto_compose(composer, output_prefix)
    
    # 评估结果
    setup_logging()
    output_prefix = os.path.join("output", model_name)
    cluster_params_string = ",".join([f"{k}: {v}" for k, v in cluster_params.items()])
    evaluate_specific_output(output_prefix, cluster_params_string)