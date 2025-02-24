# pyEnergy - 电力负荷分解与分析工具

## 项目简介

pyEnergy是一个用于电力负荷分解和分析的Python工具包。它能够对电力负荷数据进行聚类分析、信号分解和特征提取，帮助理解和优化用电模式。

## 主要功能

- 电力负荷数据预处理和导入
- 负荷特征提取和选择
- 多种聚类算法支持（K-means、密度聚类等）
- 信号分解和重构
- 可视化分析工具

## 安装说明

### 环境要求

- Python 3.12
- 依赖包：numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, pulp

### 使用Conda安装

```bash
# 克隆项目后，在项目根目录执行：
conda env create -f environment.yml

# 激活环境
conda activate nilm312
```

## 使用示例

### 1. 数据导入和预处理

```python
from pyEnergy import check

# 导入变压器数据
data_files = "path/to/your/data.csv"
df = check.import_transformer_data(data_files)
```

### 2. 负荷分解

```python
from pyEnergy.composition import composition

# 创建composer实例并设置参数
composer = composition.Composer(fool, y_pred)
composer.set_param("realP_B")

# 执行自动分解
composition.auto_compose(composer, "output/result_prefix")
```

## 项目结构

```
pyEnergy/
├── CONST.py          # 常量定义
├── check.py          # 数据检查工具
├── compute.py        # 计算功能
├── drawer.py         # 绘图工具
├── clusters/         # 聚类算法
├── composition/      # 负荷分解
└── preprocess/       # 数据预处理
```

## 输出说明

分析结果将生成以下文件：
- `*_error.csv`: 记录每个事件的平均误差
- `*_signal{i}of{n}.csv`: 分解后的各个组件信号

## 注意事项

- 输入数据需要符合指定的时间格式（支持'%Y-%m-%d %H:%M:%S'或'%m.%d.%y %H:%M'）
- 确保数据质量，去除重复值和异常值
- 大规模数据处理时注意内存使用