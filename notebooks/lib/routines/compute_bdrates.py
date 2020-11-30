import numpy as np, pandas as pd, matplotlib.pyplot as plt

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
# beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
# if not 'nb_dir' in globals():
#     nb_dir = os.getcwd()

# # #load the libraries
# from lib import *

# %autocall 1
# %load_ext autoreload
# %autoreload 2

#Goal: fix compute_bdrates_from_log_w_subsampling to compute from times instead of indices.

def compute_bdrates_from_log_w_mean(src, min_time_to_next_event = 0, min_time = 100):
    '''returns pandas.DataFrame instance with the mean rate for a given change in tip number and tip number.
    min_time_to_next_event is in units of milliseconds.  
    min_time is the time when bd events stop being considered garbage in milliseconds.
    src is a .csv filename.'''
    df = pd.read_csv(src)
    df = df[df.t>=min_time].copy()
    df = df.groupby('t').n.mean().reset_index()
    #compute birth death rates
    df['dn'] = df.n.diff().shift(-1)
    df = df.query('dn != 0').copy()
    # #extract the timeseries of tip number
    # n_series = df.n#df[boo].n
    # t_series = df.t#df[boo].t

    time_to_next_event = df.t.diff().shift(-1).dropna().values
    event_times = df.t.values
    dn_lst = df.dn.values
    
    #iterate through times, if time_to_next_event[j] is too small, then merge add it to the bucket
    time_to_next_event = list(df.t.diff().shift(-1).dropna().values)
    n_lst = list(df.n.values)
    event_times = list(df.t.values)

    #pop the first entry
    t0 = time_to_next_event.pop(0)
    n0 = n_lst.pop(0)

    bucket = 0.
    dn_lst_out = []
    n_lst_out = []
    rate_lst = []
    time_lst = []
    time = min_time
    for j, wait in enumerate(time_to_next_event):
        n1 = n_lst[j]
        time   += wait
        bucket += wait

        #if the event took enough time
        if (wait>min_time_to_next_event):
            #empty the bucket
            rate = 1/bucket
            t0 = time-bucket
            bucket = 0.

            #and append the rate
            rate_lst.append(rate)

            #compute the change in n. Append
            dn = n1 - n0
            n_lst_out.append(n0) #number of tips at beginning of transition
#             n_lst_out.append(n1) #number of tips at end of transition
            dn_lst_out.append(dn)
            n0 = n1
            time_lst.append(t0)

    dct = {'dn':dn_lst_out,'rate':rate_lst, 'n':n_lst_out,'time':time_lst}
    gf = pd.DataFrame(dct)
    df_output = gf.groupby(['dn','n']).mean().reset_index()
    return df_output, gf


def compute_bdrates(n_series,t_series):
    df = pd.DataFrame({"t":t_series.values,"n":n_series.values})
    #compute birth death rates
    df['dn'] = df.n.diff().shift(-1)
    df = df.query('dn != 0').copy()
    rates = 1/df['t'].diff().shift(-1).dropna() # birth death rates in unites of 1/ms
    df['rates'] = rates
    # df.dropna(inplace=True) #this gets rid of the termination time datum.  we want that!
    df.index.rename('index', inplace=True)
    return df

def birth_death_rates_from_log(input_file_name, data_dir_bdrates, 
                               col_n = 'n', col_t = 't', 
                               kill_all_odd_rows = True, 
                               min_time = 1000, printing = True, **kwargs):
    df = pd.read_csv(input_file_name)

    if kill_all_odd_rows:
        df.drop(df[df[col_n]%2==1].index, inplace=True)
        assert(~(df[col_n]%2==1).values.any())
    boo = df[col_t]>=min_time
    df = df[boo]

    n_series = df[col_n]
    t_series = df[col_t]

    any_tips_observed = (n_series > 0).any()

    #if there were not any tips observed, don't make a .csv in bdrates and return False
    if not any_tips_observed:
        if printing:
            print('no birth-death event was detected!')
        return False
    else:
        #store as a pandas.DataFrame
        df = compute_bdrates(n_series,t_series)
        df.to_csv(data_dir_bdrates, index=False)
        return True

# ##############################################################
# # Example Usage
# ##############################################################
# input_file_name = "/Users/timothytyree/Documents/GitHub/care/notebooks/Figures/methods/example_birth_deaths_ic_200x200.120.32_t_0_6e+03.csv"
# data_dir_bdrates = "/Users/timothytyree/Documents/GitHub/care/notebooks/Figures/methods/example_bdrates_ic_200x200.120.32_t_0_6e+03.csv"
# retval = birth_death_rates_from_log(input_file_name, data_dir_bdrates)
