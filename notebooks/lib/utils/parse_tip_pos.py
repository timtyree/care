#parse_tip_pos.py
#Programmer: Tim Tyree
#Date: 7.5.2022
#for parsing fortranic data from WJ
import re, pandas as pd, numpy as np
from itertools import zip_longest

#######################################
# Parse positions from fortranic cache
#######################################
def strip_line_to_csv(line):
    """
    Example Usage:
str_csv=strip_line_to_csv(line)
value_tuple=eval(str_csv)
    """
    str_csv=re.sub(r'\s+', ',', line.strip())
    return str_csv

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parse_fortranic_tip_pos(input_dir):
    """
    Example Usage: parse one of WJ's files to a folder of .parquet files containing spiral tip locations.
input_dir='/Users/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001'
df_log=parse_fortranic_tip_pos(input_dir)
#partition df_log into a folder of tip logs
log_folder_parquet='/home/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001_log/'
# save_df_to_parquet_by(df_log,log_folder_parquet,by='trial_num',compression='snappy',index=True)
save_df_to_parquet_by(df_log,log_folder_parquet,by='trial_num',compression='snappy',index=None)
print(f"saved to spiral tip positions to\n{log_folder_parquet=}")
#determine width and height of the computational domain input the discretization
width,height=df_log.describe().loc['max'][['x','y']].values.T
DT=g['t'].min()
printing=True
if printing:
    print(df_log.describe().loc[['min','max']][['x','y']])
    print(f"{width=}, {height=}, {DT=}")
    """
    #read in fortranic spiral tip locations
    trial_num=0
    dict_pos_lst=[]
    with open(input_dir) as f:
        for lines in grouper(f, 3, ''):
            assert len(lines) == 3
            #process lines
            str_csv=strip_line_to_csv(lines[0])
            value_tuple=eval(str_csv)
            t=value_tuple[0]
            n=value_tuple[1]
            row3=value_tuple[2]
            if n>0:
                str_csv=strip_line_to_csv(lines[1])
                x_tuple=eval(str_csv)
                str_csv=strip_line_to_csv(lines[2])
                y_tuple=eval(str_csv)
                #record for each position
                for x,y in zip(x_tuple,y_tuple):
                    dict_pos=dict(trial_num=trial_num,t=t,n=n,x=x-1,y=y-1) #-1 makes wj's min position 0 and max ~ width, as i have been using
                    dict_pos_lst.append(dict_pos)
            else:
                trial_num+=1
    df_log=pd.DataFrame(dict_pos_lst)
    return df_log


############################
# Parse positions from csv
############################
def load_tip_pos_from_csv(input_dir,round_t_to_n_digits=7,t_col='t',
                          reset_index=True,printing=True,**kwargs):
    """
    Example Usage:
df=load_tip_pos_from_csv(input_dir,round_t_to_n_digits=7,printing=True)
    """
    df=pd.read_csv(input_dir)
    df[t_col]=np.around(df[t_col],round_t_to_n_digits)
    if printing:
        print(f"before drop_duplicates: {df.shape=}")
    df.drop_duplicates(inplace=True)
    if printing:
        print(f"after drop_duplicates: {df.shape=}")
    if reset_index:
        df.reset_index(inplace=True)
    #subtract off the minimum time
    df[t_col]-=df[t_col].min()
    DT=sorted(df['t'].drop_duplicates().values)[1]
    df[t_col]+=DT
    df[t_col]=np.around(df[t_col],round_t_to_n_digits)
    boo=df['n']==0
    if sum(boo)==1:
        df=df[~boo].copy()
    return df
