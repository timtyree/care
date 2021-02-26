import numpy as np, pandas as pd
from . import *

def check_for_any_jumps (df,DT, L, DS=0.025, jump_thresh=20,drop_jumps=True,printing=False,**kwargs):
    any_jumps=False
    pid_lst=list(set(df.particle.values))
    for pid in pid_lst:
        d=df[df.particle==pid]
        jump_index_array, spd_lst = find_jumps(
            x_values=d.x.values,y_values=d.y.values,
            width=L,height=L,DS=DS,DT=DT,jump_thresh=jump_thresh)
        nj=len(jump_index_array)
        if nj>0:
            any_jumps=True
            if printing:
                print(f"the number of jumps in particle # {pid} was {nj}")
            if drop_jumps:
                df.drop(index=d.index,inplace=True)
    return any_jumps

def aggregate_all_long_traj_in_folder(input_fn_lst,T_min=1000,L=200,DS=0.025,DT=2.,
                                      num_individuals_thresh=1,omit_time=150,
                                      jump_thresh=20,drop_jumps=True,printing=False,**kwargs):
    '''returns a DataFrame of all trajectories that last longer than T_min, in time units of field t.
    run time ~ 8min for ~350 files'''
    df_lst=[]
    N_lst=[]
    DT_lst=[]
    pid_counter=0
    for fn in input_fn_lst:
        df=pd.read_csv(fn)
#         DT=compute_time_between_frames(df);#print(f"DT={DT}")#might cause problem in output
        df=get_all_longer_than(df,DT,T_min=T_min)
        #count remaining individuals
        num_individuals=len(list(set(df.particle.values)))
        if num_individuals>=num_individuals_thresh:
            #drop columns that won't be used
            column_drop_list=list(set(df.columns.values).difference(['frame','x','y','t','particle']))
            df.drop(columns=column_drop_list,inplace=True)
            #ensure all trajectories are uniquely identified by the 'particle' field
            pid_lst=list(set(df.particle.values))
            pid_max=np.max(pid_lst)
            df['particle']+=pid_counter
            pid_counter+=pid_max+1
            #remove any trajectories that jump
            boo= check_for_any_jumps (df,DT, L, DS=DS, jump_thresh=jump_thresh,drop_jumps=drop_jumps,printing=printing)
            #count remaining individuals
            num_individuals=len(list(set(df.particle.values)))
            if num_individuals>=num_individuals_thresh:
                df
                #append to lists
                N_lst.append(num_individuals)
                DT_lst.append(DT)
                df_lst.append(df)

    traj=pd.concat(df_lst)
    del df_lst
    print(f"final pid_counter value is {pid_counter}")
    print (f'total number of individuals is {sum(N_lst)}')
    print (f"constant DT? {(np.array(DT_lst)==DT_lst[0]).all()}")
    print(pid_counter)
    return traj

if __name__=='__main__':
    import sys,os
    for file in sys.argv[1:]:
        input_fn_lst = get_all_files_matching_pattern(file,trgt='_unwrap.csv')
        print(len(input_fn_lst))
        omit_time=0#ms
        DS=0.025#cm/pixel
        diffCoef=0.0005
        L=200#pixels
        DT=2.
        T_min=500#ms
        num_individuals_thresh=1
        jump_thresh=50

        traj=aggregate_all_long_traj_in_folder(input_fn_lst,T_min=T_min,L=L,DS=DS,DT=DT,
                                      num_individuals_thresh=num_individuals_thresh,omit_time=omit_time,
                                      jump_thresh=jump_thresh,drop_jumps=True,printing=True)#,**kwargs)

        #save output
        save_folder=os.path.dirname(os.path.dirname(file))
        output_file_name = f"all_traj_longer_than_{T_min}.csv"
        os.chdir(save_folder)
        traj.to_csv(output_file_name, index=False)

        print(f"output saved in {os.path.abspath(output_file_name)}")
