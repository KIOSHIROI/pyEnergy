import pandas as pd
import valuation as val
import numpy as np
from pyEnergy.composition.composition import Composer
from pyEnergy.fool import Fool
df_val = val.get_val_data()


path = "data/ChangErZhai-40-139079-values 20180101-20181031.csv"
fool = Fool(path).select(method='pca', threshold=0.3, n_components=5)
from pyEnergy.clusters.HAC import HAC
y_pred = HAC(fool).fit()
composer = Composer(fool, y_pred, threshold=1)
composer.set_param('realP_B')
composer.set_reducer('my3')
df_pred = pd.DataFrame({'UTC Time':[], 'workingPower':[]})
for i in range(0, 1):
    composer.compose(index=i)
    df_pred = pd.concat([df_pred, composer.pred_signal]).drop_duplicates(subset='UTC Time')
rmse = val.get_rmse(df_val=df_val, df_pred=df_pred)
print(rmse)
