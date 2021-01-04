import pandas as pd, numpy as np, os
# Tim Tyree
# 11.21.2020

#Old way
# def log_to_bdrates(input_file_name, output_file_name):
#     '''imports a raw output tip log file from input_file_name and saves to a birth-death rate file in output_file_name. '''
#     df = pd.read_csv(input_file_name)

#     #extract n_series
#     n_list = []
#     for i, row in df.iterrows():
#         n = len(eval(row.x))
#         n_list.append(n)
#     df['n'] = n_list
#     n_series = df.n
#     n_series.index = df.t

#     #compute birth-death rates
#     #store as a pandas.DataFrame
#     df = pd.DataFrame({"t":n_series.index.values,"n":n_series.values})

#     #compute birth death rates
#     df['dn'] = df.n.diff().shift(-1)
#     df = df.query('dn != 0').copy()
#     rates = 1/df['t'].diff().shift(-1).dropna() # birth death rates in unites of 1/ms
#     df['rates'] = rates

#     #save birth death rates to a file named according to all of the relevant parameters in a special folder.
#     df.to_csv(output_file_name)
#     return True

def log_to_bdrates_routine(input_file_name, save_folder, min_time_between_samples, output_file_name=None):
    '''input_file_name contains the log of spiral tips.  birth-death rates are computed and saved to output_file_name.
    subsampling is used, such that the time between two frames is no less than min_time_between_samples'''
    #compute the input/output absolute paths
    #     src = os.path.join(folder_name,os.path.basename(input_file_name))
    if output_file_name is None:
        output_file_name = os.path.basename(input_file_name).replace('log.csv','bdrates.csv')
    dst = os.path.join(save_folder,output_file_name)    
    df = compute_bdrates_from_log_w_subsampling(input_file_name,min_time_between_samples=min_time_between_samples)

    #save birth death rates to a file named according to all of the relevant parameters in a special folder.
    df.to_csv(dst, index=False, encoding="utf-8")
    return dst

def compute_bdrates_from_log_w_subsampling(src,min_time_between_samples, **kwargs):
    df = pd.read_csv(src)

    #subsample tip log rows to mimick lower sampling rates
    times_raw = sorted(set(df.t.values))
    times_filtered = []
    time_counter = min_time_between_samples
    while len(times_raw)>1:
        DT = times_raw[1]-times_raw[0] #not supposing constant sampling period
        t = times_raw.pop(0)
        time_counter += DT
        if time_counter >= min_time_between_samples:
            times_filtered.append(t)
            time_counter = 0

    #make a boolean index for these times_filtered
    boo = df.t==-9999
    for t in times_filtered:
        boo |= df.t==t #np.isclose(df.t,t)

    #extract the timeseries of tip number
    n_series = df[boo].n
    t_series = df[boo].t

    #compute birth-death rates
    #store as a pandas.DataFrame
    df = pd.DataFrame({"tid":n_series.index.values,"n":n_series.values})


    #compute birth death rates
    df['dn'] = df.n.diff().shift(-1)
    df = df.query('dn != 0').copy()

    # TODO: check that df is sorted at this line.

    rates = 1/df['tid'].diff().shift(-1).dropna() # birth death rates in unites of 1/ms
    df['rates'] = rates

    return df

def produce_one_csv(list_of_files, file_out):
   # Consolidate all csv files into one object
   result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")


# ############################################################################### #
# ### Example Usage: compute birth-death rates for whole folder of log files  ### #
# ############################################################################### #
# #get a folder of filenames of tip log files ending in log.csv
# #find file interactively
# print("please select a file from within the desired folder.")
# file = search_for_file()
# folder_name = os.path.dirname(file)
# os.chdir(folder_name)
# # get all .csv files in the current working directory
# retval = !ls
# file_name_list = list(retval)
# # check each file if it ends in .csv before merging it
# def is_target(file_name, target = '_log.csv'):
#     return file_name[-len(target):]==target
# file_name_list = [os.path.join(folder_name,f) for f in file_name_list if is_target(f, target = 'log.csv')]
# min_time_between_samples = 20 #milliseconds
# #create a save_folder, for bdrates if it doesn't already exist
# base_folder = os.path.dirname(folder_name)
# save_folder_name = os.path.join(base_folder,f'birth-death-rates-sampled-every-{min_time_between_samples}-ms')
# save_folder = os.path.join(base_folder,save_folder_name)
# if not os.path.exists(save_folder):
#     os.chdir(base_folder)
#     os.mkdir(save_folder_name)
#     print(f"created save folder at {save_folder}.")
# #run routine
# print('this should be an absolute path:')
# print(file_name_list[0])
# output_file_name_list = []
# for input_file_name in file_name_list:
#     outdir = log_to_bdrates_routine(input_file_name, save_folder, min_time_between_samples)
#     output_file_name_list.append(outdir)
# #consolodate rates into one csv
# # file_out = "../consolidated_rates.csv"
# os.chdir(save_folder)
# file_out = f"../consolidated_rates-sampled-every-{min_time_between_samples}-ms.csv"
# produce_one_csv(list_of_files=output_file_name_list, file_out=file_out)

# print(len(output_file_name_list))
