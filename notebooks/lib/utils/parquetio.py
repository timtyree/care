#parquetio.py
#Programmer: Tim Tyree
#Date: 6.29.2022
import shutil, os, pandas as pd, numpy as np
def save_df_to_parquet_by(df,folder_parquet,by='trial_num',
                          compression='snappy',index=None,
                          **kwargs):
    """save_df_to_parquet_by groups pd.DataFrame instance, df by by='trial_num',
    and then saves each group into a separate parquet file in

    options for compression: {'snappy', 'gzip', 'brotli', None}, default 'snappy'
    see df.to_parquet? for details on compression

    Example Usage:
log_folder_parquet='/home/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001_log/'
save_df_to_parquet_by(df_log,log_folder_parquet,by='trial_num',compression='snappy',index=None)
    """
    #clear cache before saving into it
    if os.path.exists(folder_parquet):
        shutil.rmtree(folder_parquet,ignore_errors=True)
    #save to parquet
    df.to_parquet(folder_parquet,index=index,
                     compression=compression,
                      partition_cols=[by])
    return True

def load_parquet_by_trial_num(trial_num,folder_parquet,reset_index=True,**kwargs):
    """
    Example Usage:
log_folder_parquet='/home/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001_log/'
g=load_parquet_by_trial_num(trial_num=639,folder_parquet=log_folder_parquet)
    """
    log_dir_parquet=os.path.join(folder_parquet,f"{trial_num=}")
    df=pd.read_parquet(log_dir_parquet)
    df['trial_num']=trial_num
    if reset_index:
        df.reset_index(inplace=True)
    #set index dtyp
    #df.index = df.index.map(str)
    #df.index = df.index.map(np.int64)
    df.index = df.index.astype(np.int64)
    return df
