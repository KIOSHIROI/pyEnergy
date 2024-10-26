1. cumpute.py 中的 standard()更名为z_score()，normalize()更名为min_max()，更加直观易懂。
2. 将 fool.py 中 Fool 中的标准化从min_max改为z_score
3. 移除 core.py 中默认的特征列表。新的特征列表由compute()函数内使用的特征自动生成。
4. 新建feature_computer.py 将core中的compute_features()移除并并更新其他包的引用。增加drought season的计算。drought season的定义根据华北平原降水量和灌溉习惯定义为2-6月.drought season计算接口为compute_season(). 增加compute_features()的参数drought.如果drought=True, 特征将仅计算旱季的数据。
5. 在final.py中更新了新的平滑方法my_reduce().my_reduce的计算规则为先对大小为三的滑动块内的数据进行计算，如果中间数据为突变信号，则将信号替换为左右数据平均值。再对大小为四的华东快内的数据进行计算，如果中间两个数据为突变信号，则将信号替换为左右数据的平均值。这种平滑方法能够有效处理短时突变信号而对信号边界有较小影响。