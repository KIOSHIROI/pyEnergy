from matplotlib import pyplot as plt
import pyEnergy.compute as cm
import pyEnergy.drawer as dw
import os
df_val = cm.interpolation_val()
dw.plot_signal([
    cm.interpolation_pred(r"output\Kmeans\autoencoder30\Kmeans_si_pca8_signal2of3.csv", df_val),
    df_val
    ])
plt.show()