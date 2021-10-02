import numpy as np, cudf, cupy as cp 
def get_DT_cu(df,t_col='t',pid_col='particle',round_digits=7):
    pid_values=df[pid_col].drop_duplicates().values
    return float(np.around(df.loc[pid_values[0]==df[pid_col],t_col].diff().tail(1).values.get(),round_digits))
