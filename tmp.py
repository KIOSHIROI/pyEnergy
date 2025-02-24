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
output_prefix = os.path.join("output","HAC_two_stage_w10")
cluster_params = "repeats: 20, weights: [1, 0]"
evaluate_specific_output(output_prefix, cluster_params)