import datetime
import numpy as np


feature_info = [
    'std. real power(ss)', 
    'ave. real power(ss)', 
    'trend real power(ss)', 
    'max. real power(tr)',

    'std. reactive power(ss)', 
    'ave. reactive power(ss)', 
    'trend reactive power(ss)', 
    'max. reactive power(tr)',
    
    'std. phase B current(ss)', 
    'ave. phase B current(ss)', 
    'trend phase B current(ss)', 
    'max. phase B current(tr)',
    ''
]

param_feature_dict = {
    "curnt_B": ['std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)'],
    "realP_B": ['std. real power(ss)', 'ave. real power(ss)', 'max. real power(tr)'],
    "reactP_B": ['std. phase B current(ss)', 'ave. phase B current(ss)', 'max. phase B current(tr)']
}

REAL_POWER_FEATURES = [
    'std. real power(ss)',
    'ave. real power(ss)',
    'trend real power(ss)',
    'max. real power(tr)'
]

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