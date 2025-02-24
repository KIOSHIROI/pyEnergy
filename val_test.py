from matplotlib import pyplot as plt
import pyEnergy.compute as cm
import pyEnergy.drawer as dw
import os
# dw.plot_signal([cm.interpolation_val()])
# plt.show()
dir_path = "output/Kmeans/autoencoder30/"
for pred_path in os.listdir(dir_path):
    if pred_path.split('_')[-1] == 'error.csv':
        continue
    path = dir_path + pred_path
    df_val = cm.interpolation_val()
    df_pred = cm.interpolation_pred(path, df_val)
    inclution = cm.compute_inclution(df_pred, df_val)
    matching = cm.compute_matching(df_pred, df_val)
    time_inclu = cm.compute_time_inclution(df_pred, df_val)
    time_match = cm.compute_time_matching(df_pred, df_val)
    power_errors = cm.compute_power_error_by_state(df_pred, df_val)
    
    if time_inclu > 0.6:
        print(10*'-'+f"{pred_path}"+10*'-')
        print(
            f"inclution:\t{inclution*100:6.3f}%\n"
            f"matching:\t{matching*100:6.3f}%\n"
            f"time incl:\t {time_inclu*100:6.3f}%\n"
            f"time match:\t{time_match*100:6.3f}%\n"
            f"active err:\t{power_errors['active_power_error']:6.3f}\n"
            f"inactive err:\t{power_errors['inactive_power_error']:6.3f}"
            )
print("end")







