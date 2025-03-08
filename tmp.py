import os
import datetime
import numpy as np
import logging
import pyEnergy.compute as cm
from matplotlib import pyplot as plt
import pyEnergy.drawer as dw
from pyEnergy.eval import evaluate_specific_output, setup_logging
# 初始化日志
setup_logging()

# 评估特定的输出结果
# model_and_attr = r"OLD_METHODw10_notL\two_stage_pca6_clusters10"

model_and_attr = r"OLD_METHODw10\two_stage_pca6_clusters10"

output_prefix = os.path.join("output", model_and_attr)

cluster_params = "repeats: 10, weights: [0.5, 0.5]"
evaluate_specific_output(output_prefix, cluster_params)