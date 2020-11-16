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
