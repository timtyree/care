import pandas as pd, numpy as np
# from scipy import stats
from .compute_slope import *

def compute_slope_vs_lag(emsd,T_min,omit_time,window_width=300,stepsize=10):
    tau_min_values=np.arange(0,T_min-2*omit_time,stepsize)#ms
    #compute the slope over a sliding window
    slope_lst=[]
    for tau_min in tau_min_values:
        tau_max=tau_min+window_width
        #measure the slope of the log-log plot
        tv=emsd.index.values
        boo=(tv>=tau_min)&(tv<=tau_max)
        x_values=np.log10(tv[boo])
        y_values=np.log10(emsd.values[boo])
        dict_output=compute_95CI_ols(x_values, y_values)
        slope=dict_output['m']
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        #     print(f"slope={slope:.4f} +-{std_err:.4f}, intercept={intercept:.4f}, T_min={T_min:.0f}, tau_min={tau_min:.0f}, tau_max={tau_max:.0f}, N={num_individuals}")
        slope_lst.append(slope)
    slope_values=np.array(slope_lst)
    lag_values=(tau_min_values+window_width/2)/10**3
    return lag_values,slope_values
