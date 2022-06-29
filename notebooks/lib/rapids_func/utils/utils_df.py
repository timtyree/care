# utils_df.py
#Programmer: Tim Tyree
#Date: 6.29.2022
import shutil, os, sys, pandas as pd, cudf
def copy_df_as_pandas(df):
    """copy_df_as_pandas takes pandas.DataFrame and cudf.DataFrame instances alike and returns a copy of the df as a pandas.DataFrame (perhaps useful good for saving)."""
    if str(type(df))=="<class 'cudf.core.dataframe.DataFrame'>":
            df_=df.to_pandas()
    else: #if str(type(dfr))=="<class 'pandas.core.frame.DataFrame'>":
        df_=df.copy()
    return df_

def comp_lifetimes_by(df,t_col='t',by='particle',pid_lst=None,printing=False,**kwargs):
    """
    Example Usage:
df_lifetimes_dfr=comp_lifetimes_by(df=dfr.to_pandas(),t_col='t',by='pid_other',pid_lst=None,printing=True)#,**kwargs)
    """
    dft=df.reset_index().groupby([by]).describe()[t_col]
    df_lifetimes=-dft[['max','min']].T.diff().loc['min']
    #DONE: print the lifetime of all pid_lst_stumps
    if printing:
        if pid_lst is None:
            pid_lst=list(np.unique(df[by].values))
        for i,pid in enumerate(pid_lst):
            try:
                print(f"apparent lifetime of particle #{pid}:\t{df_lifetimes.loc[pid]} ms.")
            except Exception as e:
                print(f"Warning: {e}")
    return df_lifetimes
