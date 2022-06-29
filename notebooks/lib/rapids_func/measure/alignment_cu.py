#alignment_cu.py
#Programmer: Tim Tyree
#Date: 6.29.2022
import shutil, os, pandas as pd, numpy as np, cudf, cupy
from ...measure.compute_slope import comp_ols_simple

def align_timeseries_simple(dfr,df_pairs,
                            P_col='index_pairs',
                            R_col='R_nosavgol',
                            T_col='tdeath',
                            T_col_out='talign',
                            max_num_obs_align=3,
                            **kwargs):
    """align_timeseries_simple shifts the times by tshift, determined by linear estimation of the true extremal time.

    Example: Usage:
df_R,df_P=align_timeseries_simple(dfr,df_pairs,
                            P_col='index_pairs',
                            R_col='R_nosavgol',
                            T_col='tdeath',
                            T_col_out='talign',
                            max_num_obs_align=3)
    """
    if str(type(dfr))=="<class 'cudf.core.dataframe.DataFrame'>":
        df_R=dfr.to_pandas()
    else: #if str(type(dfr))=="<class 'pandas.core.frame.DataFrame'>":
        df_R=dfr.copy()
    if str(type(dfr))=="<class 'cudf.core.dataframe.DataFrame'>":
        df_P=df_pairs.to_pandas()
    else: #if str(type(dfr))=="<class 'pandas.core.frame.DataFrame'>":
        df_P=df_pairs.copy()

    #initialize virtual memory
    df_R[T_col_out]=0.
    for index_pairs,row in df_P.iterrows():
        df=df_R[df_R[P_col]==index_pairs]
        d=df.sort_values(by=T_col).head(max_num_obs_align)
        x_values=d[R_col].values
        y_values=d[T_col].values

        #compute slope from x_values,y_values
        #get the y-intercept as -tshift
        if x_values.shape[0] > 2:
            # tshift=-y_intercept_of_ols
            tshift=-comp_ols_simple(x_values,y_values)['b']
        else:
            if (x_values.shape[0] < 2):
                #tshift= NaN
                tshift=np.nan
            elif (x_values[1]>x_values[0]):
                # tshift=-y_intercept_of_linear_solution
                b=y_values[0]-x_values[0]*(y_values[1]-y_values[0])/(x_values[1]-x_values[0])
                tshift=-b
            else:
                #annihilating positions only move towards each other
                #tshift= NaN
                tshift=np.nan
        #record tshift
        df_R.loc[df_R['index_pairs']==index_pairs,'talign']=tshift
    #compute tdeath_align
    df_R['tdeath_align'] = df_R['tdeath'] + df_R['talign']
    # df_R['talign'].plot.hist(bins=300)
    return df_R,df_P
