from matplotlib import pyplot as plt
import pyEnergy.eval as cm
import pyEnergy.drawer as dw
import os
df_val = cm.interpolation_val()
dw.plot_signal([
    cm.interpolation_pred(r"output\OLD_METHOD5511\data\signal1of4.csv", df_val),
    # cm.interpolation_pred(r"output\OLD_METHODw10_notL\two_stage_pca6_clusters10\data\_signal10of10.csv", df_val),
    
    df_val
    ], labels=["平滑后",
    #  "平滑前",
      "val"])
plt.show()