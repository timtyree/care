# Tim Tyree
# 12.18.2020

import trackpy, pandas as pd, numpy as np
from .. import *
from .track_tips import *
from ..utils.dist_func import *
from ..utils.utils_traj import *

def preprocess_log(input_file_name):
    '''prep and filters raw trajectory
    output_file_name = preprocess_log(input_file_name)
    '''
    #track tips for given input file

    output_file_name = generate_track_tips_pbc(input_file_name, save_fn=None)
    return output_file_name

def unwrap_trajectories(input_file_name, output_file_name, width, height, DS, **kwargs):
    # load trajectories
    df = pd.read_csv(input_file_name)
    pid_lst = sorted(set(df.particle.values))
    #(duplicates filtered earlier_ _  _ _ ) filter_duplicate_trajectory_indices is slow (and can probs be accelerated with a sexy pandas one liner)
    pid_lst_filtered = pid_lst#filter_duplicate_trajectory_indices(pid_lst,df)
    # pid_lst_filtered = filter_duplicate_trajectory_indices(pid_lst,df)
    df = pd.concat([unwrap_traj_and_center(df[df.particle==pid], width, height, DS, **kwargs) for pid in pid_lst_filtered])
    #save results
    # dirname = os.path.dirname(input_file_name).split('/')[-1]
    dirname='trajectories'
    folder_name=os.path.dirname(input_file_name)
    save_folder = folder_name.replace(dirname,'trajectories_unwrap')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    os.chdir(save_folder)
    df.to_csv(output_file_name,index=False)
    return os.path.abspath(output_file_name)

# ####################################
# # Example Usage
# ####################################
# input_file_name = "/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ds_5_param_set_8_fastkernel_V_0.4_archive/Log/ic_200x200.001.11_log.csv"
# output_file_name = preprocess_log(input_file_name)
#
#
# #select the longest n trajectories
# n_tips = 1#15
# s = df.groupby('particle').t.count()
# s = s.sort_values(ascending=False)
# pid_longest_lst = list(s.index.values[:n_tips])
# # d = df[df.particle==pid_longest]
# # print(pid_longest)
# # print(s.head())
# # pid_longest_lst = s.head(n_tips).values
# df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])
# assert ( (np.array(sorted(set(df_traj['particle'].values)))==np.array(sorted(pid_longest_lst))).all())
#





# #tests
# d = unwrap_traj_and_center(d).copy()
#
# #test that unwrap_traj_and_center removed all jump detections
# x_values = d.x.values.astype('float64')
# y_values = d.y.values.astype('float64')
# jump_index_array, spd_lst = find_jumps(x_values,y_values)
# assert (jump_index_array.size==0)

# def compute_emsd_for_longest_trajectories(input_file_name,n_tips = 1,DS = 5/200,DT = 1., round_t_to_n_digits=0):
def get_longest_trajectories(input_file_name, width, height, n_tips = 1,DS = 5/200,DT = 2., round_t_to_n_digits=0, jump_thresh=20., **kwargs):
    #select the longest trajectories that moves
    df = pd.read_csv(input_file_name)
    df.reset_index(inplace=True)
    if df.x.values.shape[0]==0:
        return None
    if n_tips==1:
        try:
            s = df.groupby('particle').t.count()
        except KeyError as e:#("KeyError") as e:
            # print(e)
            print( f"\t trial that failed: {input_file_name.split('/')[-1]}")
            return None
        s = s.sort_values(ascending=True)
        pid_longest_lst = list(s.index.values)#[:n_tips])
        pid=pid_longest_lst.pop()
        std_diffx=df[(df.particle==pid)].x.diff().dropna().std()
        boo=False
        if std_diffx:#.diff().dropna().x.std
            if std_diffx>0:
                boo=True
        while not boo:
            pid=pid_longest_lst.pop()
            std_diffx=df[(df.particle==pid)].x.diff().dropna().std()
            boo=False
            if std_diffx:#.diff().dropna().x.std
                if std_diffx>0:
                    boo=True
        pid_longest_lst=[pid]
    else:
        try:
            s = df.groupby('particle').t.count()
        except KeyError as e:#("KeyError") as e:
            # print(e)
            print( f"\t trial that failed: {input_file_name.split('/')[-1]}")
            return None
        s = s.sort_values(ascending=False)
        if s.index.values.shape[0]==0:
            return None
        pid_longest_lst = list(s.index.values[:n_tips])
    #     df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])

    #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    df_lst = []
    for pid in  pid_longest_lst:#[2:]:
        d = df[(df.particle==pid)].copy()
        x_values, y_values = d[['x','y']].values.T
        index_values = d.index.values.T
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height, DS=DS,DT=DT, jump_thresh=jump_thresh, **kwargs)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)

    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    # df_traj['x'] = df_traj.x.round(6)
    # df_traj['y'] = df_traj.y.round(6)
    try:
        df_traj['grad_ux'] = df_traj.grad_ux.round(5)
        df_traj['grad_uy'] = df_traj.grad_uy.round(5)
        df_traj['grad_vx'] = df_traj.grad_vx.round(5)
        df_traj['grad_vy'] = df_traj.grad_vy.round(5)
    except AttributeError as e:
        #the trajectories are apparently missing field gradients...
        pass
    #     assert ( (np.array(sorted(set(df_traj['particle'].values)))==np.array(sorted(pid_longest_lst))).all())
    return df_traj

def compute_emsd_for_longest_trajectories(input_file_name,n_tips,DS,DT, L, round_t_to_n_digits=0, **kwargs):
    df_traj=get_longest_trajectories(input_file_name,n_tips=n_tips,DS=DS,DT=DT,width=L,height=L,round_t_to_n_digits=round_t_to_n_digits)
    try:
        #compute ensemble mean squared displacement
        emsd = trackpy.motion.emsd(df_traj, mpp=1., fps=1.,max_lagtime=40000)
        #cast ensemble mean squared displacement into units of cm^2 and seconds
        return pd.DataFrame({'msd':DS**2*emsd.values, 'lagt':emsd.index.values*DT/10**3, 'src':os.path.basename(input_file_name)})
    except ValueError as e:#ValueError: No objects to concatenate
        print("ValueError: No objects to concatenate")
        print(f"\ttrial that failed: {input_file_name.split('/')[-1]}")
        return None

def compute_average_msd(df, DT):
    src_lst = sorted(set(df.src.values))
    src_lst = src_lst#[:10]
    ff = df.copy()#pd.concat([df[df.src==src] for src in src_lst])
    dt = DT/10**3
    t_values = np.array(sorted(set(ff.lagt.values)))
    t_values = np.arange(np.min(t_values),np.max(t_values),dt)
    # averaging msd over trials
    msd_lst = []
    for t in t_values:
        boo = (ff.lagt>=t-dt/2)&(ff.lagt<=t+dt/2)
        msd_lst.append(ff[boo].msd.mean())
    t_values = t_values
    msd_values = np.array(msd_lst)
    return t_values, msd_values

def compute_average_std_msd(df,DT):
    '''df is a pandas.DataFrame instance that has fields src, lagt, and msd'''
    src_lst = sorted(set(df.src.values))
    ff = df.copy()#pd.concat([df[df.src==src] for src in src_lst])
    dt = DT/10**3 #seconds per frame
    t_values = np.array(sorted(set(ff.lagt.values)))
    t_values = np.arange(np.min(t_values),np.max(t_values),dt)
    # averaging msd over trials
    msd_lst = []
    std_lst = []
    for t in t_values:
        #this binning is robust to floating point error
        boo = (ff.lagt>=t-dt/2)&(ff.lagt<=t+dt/2)
        msd_vals=ff[boo].msd
        msd_lst.append(msd_vals.mean())
        std_lst.append(msd_vals.std())
    t_values = t_values
    msd_values = np.array(msd_lst)
    std_values = np.array(std_lst)
    return t_values, msd_values, std_values

def PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,0.05],ylim=[0,4],D = 75,saving = True,fontsize =22,figsize=(9,6),
    use_ylim=False,use_xlim=True,**kwargs):
    '''plot msd for each trial listed in df.src. also plot average msd.'''
    #compute average msd by trial for a subset of trials
    fig, ax = plt.subplots(figsize=figsize)
    # x_values = ef.index.values#*DT/10**3 #lag in seconds
    # y_values = ef.values#*DS**2 #msd
    src_lst = sorted(set(df.src.values))
    for src in src_lst:
        x_values = df[df.src==src].lagt.values
        y_values = df[df.src==src].msd.values
        ax.plot(x_values,y_values,c='blue',alpha=0.2)
    ax.fill_between(t_values,msd_values-std_values,msd_values+std_values,color='green',lw=2, alpha=0.3)
    ax.plot(t_values,msd_values,c='g',lw=2)
    ax.plot(t_values,t_values*D,c='r',lw=2)

    # DS = 5/200 #cm per pixel
    # DT = 1. #ms per frame
    if use_xlim:
        ax.set_xlim(xlim)
    if use_ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel('lag (s)', fontsize=fontsize)
    ax.set_ylabel(r'MSD (cm$^2$)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    if not saving:
        plt.show()
    else:
        plt.tight_layout()
        os.chdir(savefig_folder)
        plt.savefig(savefig_fn, dpi=300)
        print(f"saved figure in \n\t{savefig_folder}")
        plt.close()
    return {'t':t_values,'msd':msd_values}

def generate_msd_figures_routine(input_file_name,n_tips, **kwargs):#V_thresh=None):
    '''file is an unwrapped trajectory csv file'''
    # if V_thresh==None:
    #     if file.find('_V_')!=-1:
    #         V_thresh = eval(file[file.find('_V_')+len('_V_'):].split('_')[0])
    # input_file_name=file
    # get all .csv files in the working directory of ^that file
    folder_name = os.path.dirname(input_file_name)
    os.chdir(folder_name)
    retval = os.listdir()#!ls
    file_name_list = list(retval)
    # check each file if it ends in .csv before merging it
    trgt='_unwrap.csv'
    def is_csv(file_name,trgt):
        return file_name[-len(trgt):]==trgt
    file_name_list = [f for f in file_name_list if is_csv(f,trgt)]
    return generate_msd_figures_routine_for_list(file_name_list,n_tips,**kwargs)

def generate_msd_figures_routine_for_list(file_name_list, n_tips,DT, DS,L, output_file_name=None,save_folder=None,**kwargs):
    '''file_name_list is a list of _unwrap.csv files.
    returns a string indicating the output_file_name beginning in emsd_...'''#, V_thresh=None
    # file=file_name_list[0]
    # folder_name = os.path.dirname(file)
    input_file_name=os.path.abspath(file_name_list[0])
    folder_name = os.path.dirname(input_file_name)
    os.chdir(folder_name)
    print(f"Num. file names in list = {len(file_name_list)}.")
    df=pd.read_csv(input_file_name)
    #compute DT explicitely
    DT = np.mean(df.t.diff().dropna().values) #ms per frame

    #compute ensemble mean squared displacement for the longest n_tips for each trial in file_name_list
    os.chdir(folder_name)
    dict_out_lst=[compute_emsd_for_longest_trajectories(input_file_name, n_tips=n_tips,DS=DS,DT=DT,L=L) for input_file_name in file_name_list]
    if len(dict_out_lst)==0:
        print(f"""no sufficiently long lasting trajectory was found.  returning None, None for
        input_file_name, {input_file_name}.""")
        return None, None
    df = pd.concat(dict_out_lst)
    df.reset_index(inplace=True,drop=True)
    #save results
    dirname = os.path.dirname(input_file_name).split('/')[-1]
    folder_name=os.path.dirname(input_file_name)
    if save_folder is None:
        save_folder = folder_name.replace(dirname,'msd')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if output_file_name is None:
        output_file_name = f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv"
    os.chdir(save_folder)
    df.to_csv(output_file_name, index=False)

    #compute average msd by trial for a subset of trials
    src_lst = sorted(set(df.src.values))
    # src_lst = src_lst#[:10]
    ff = df.copy()#pd.concat([df[df.src==src] for src in src_lst])
    # dt = DT/10**3 #seconds per frame
    # t_values = np.array(sorted(set(ff.lagt.values)))
    # t_values = np.arange(np.min(t_values),np.max(t_values),dt)#no floating point error

    t_values, msd_values, std_values = compute_average_std_msd(df,DT)
    #     t_values, msd_values = compute_average_msd(df, DT=1.)

    sl=input_file_name.split('/')
    trial_folder_name=sl[-3]


    # savefig_folder = os.path.join(nb_dir,f'Figures/msd/'+trial_folder_name)#V_{V_thresh}')
    # if V_thresh is not None:
    #     savefig_folder = os.path.join(nb_dir,f'Figures/msd/'+trial_folder_name)#V_{V_thresh}')
    # if save_folder is None:
    #     savefig_folder = 'fig'
    #     savefig_folder=os.path.abspath(savefig_folder)
    # savefig_folder=os.path.join(save_folder,'/fig')
    os.chdir(save_folder)
    savefig_folder='fig'
    if not os.path.exists(savefig_folder):
        os.mkdir(savefig_folder)
    os.chdir(savefig_folder)
    savefig_folder=os.getcwd()
    # generate plots of msd's
    savefig_fn = os.path.basename(output_file_name).replace('.csv','_long_time_std.png')
    retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,4],ylim=[0,10],saving = True,fontsize =22,figsize=(9,6),D=3.5)

    savefig_fn = os.path.basename(output_file_name).replace('.csv','_short_time_std.png')
    retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,0.2],ylim=[0,1],saving = True,fontsize =22,figsize=(9,6),D=3.5)

    savefig_fn = os.path.basename(output_file_name).replace('.csv','_very_short_time_std.png')
    retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,0.05],ylim=[0,0.2],saving = True,fontsize =22,figsize=(9,6),D=3.5)
    return output_file_name

# beep(4)

def compute_emsd_dataframe_for_trial_list(file_name_list, T_min, DT, DS=0.025,L=200,num_individuals_thresh=30, omit_time=150,**kwargs):
    '''file_name_list is a list of _unwrap.csv files.
    returns a string indicating the output_file_name beginning in emsd_...
    t_values, msd_values, std_values = compute_emsd_errorbars_btwn_trials(file_name_list, T_min, DT, DS=0.025,L=200,num_individuals_thresh=30)
    '''
    # file=file_name_list[0]
    # folder_name = os.path.dirname(file)

    input_file_name=os.path.abspath(file_name_list[0])
    folder_name = os.path.dirname(input_file_name)
    os.chdir(folder_name)
    print(f"Num. file names in list = {len(file_name_list)}.")
    # df=pd.read_csv(input_file_name)
    # #compute DT explicitely
    # # DT = np.mean(df.t.diff().dropna().values[:1]) #ms per frame
    # DT=compute_time_between_frames(df)
    #compute ensemble mean squared displacement for the longest n_tips for each trial in file_name_list
    os.chdir(folder_name)
    dict_out_lst=[]
    for input_file_name in file_name_list:
        traj=get_all_longer_than(pd.read_csv(input_file_name),DT,T_min=T_min)

        #compute number of num_individuals
        pid_lst=sorted(set(traj.particle.values))
        num_individuals=len(pid_lst)
        #if enough individuals are present in the given traj
        if num_individuals>=num_individuals_thresh:
            #compute ensemble mean squared displacement
            emsd=compute_emsd(traj=traj, DT=DT, omit_time=omit_time, printing=False,DS=DS)
            # emsd = trackpy.motion.emsd(df_traj, mpp=1., fps=1.,max_lagtime=40000)
            #cast ensemble mean squared displacement into units of cm^2 and seconds
            d=pd.DataFrame({'msd':emsd.values,
                    'lagt':emsd.index.values/10**3,
                    'src':os.path.basename(input_file_name),
                    "num_individuals":num_individuals
                })
            dict_out_lst.append(d)

    if len(dict_out_lst)==0:
        print(f"""no trial had enough individuals lasting longer than T_min. consider different parameters.  returning None for
        input_file_name, {input_file_name}.""")
        return None
    df = pd.concat(dict_out_lst)
    df.reset_index(inplace=True,drop=True)
    return df
def compute_emsd_errorbars_btwn_trials(file_name_list, T_min, DT=None, DS=0.025,L=200,num_individuals_thresh=30, omit_time=150,**kwargs):
    '''file_name_list is a list of _unwrap.csv files.
    returns a string indicating the output_file_name beginning in emsd_...
    t_values, msd_values, std_values = compute_emsd_errorbars_btwn_trials(file_name_list, T_min, DT, DS=0.025,L=200,num_individuals_thresh=30)
    '''
    if DT is None:
        input_file_name=os.path.abspath(file_name_list[0])
        folder_name = os.path.dirname(input_file_name)
        os.chdir(folder_name)
        print(f"Num. file names in list = {len(file_name_list)}.")
        df=pd.read_csv(input_file_name)
        #compute DT explicitely
        # DT = np.mean(df.t.diff().dropna().values[:1]) #ms per frame
        DT=compute_time_between_frames(df)
    df=compute_emsd_dataframe_for_trial_list(file_name_list=file_name_list, T_min=T_min, DT=DT, DS=DS,L=L,num_individuals_thresh=num_individuals_thresh, omit_time=omit_time,**kwargs)
    t_values, msd_values, std_values = compute_average_std_msd(df,DT)
    return t_values, msd_values, std_values
    # #save results
    # dirname = os.path.dirname(input_file_name).split('/')[-1]
    # folder_name=os.path.dirname(input_file_name)
    # if save_folder is None:
    #     save_folder = folder_name.replace(dirname,'msd')
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    # if output_file_name is None:
    #     output_file_name = f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv"
    # os.chdir(save_folder)
    # df.to_csv(output_file_name, index=False)

    #compute average msd by trial for a subset of trials
    # src_lst = sorted(set(df.src.values))
    # # src_lst = src_lst#[:10]
    # ff = df.copy()#pd.concat([df[df.src==src] for src in src_lst])
    # dt = DT/10**3 #seconds per frame
    # t_values = np.array(sorted(set(ff.lagt.values)))
    # t_values = np.arange(np.min(t_values),np.max(t_values),dt)#no floating point error

    # t_values, msd_values, std_values = compute_average_std_msd(df,DT)
    #     t_values, msd_values = compute_average_msd(df, DT=1.)

    # sl=input_file_name.split('/')
    # trial_folder_name=sl[-3]


    # savefig_folder = os.path.join(nb_dir,f'Figures/msd/'+trial_folder_name)#V_{V_thresh}')
    # if V_thresh is not None:
    #     savefig_folder = os.path.join(nb_dir,f'Figures/msd/'+trial_folder_name)#V_{V_thresh}')
    # if save_folder is None:
    #     savefig_folder = 'fig'
    #     savefig_folder=os.path.abspath(savefig_folder)
    # savefig_folder=os.path.join(save_folder,'/fig')
    # os.chdir(save_folder)
    # savefig_folder='fig'
    # if not os.path.exists(savefig_folder):
    #     os.mkdir(savefig_folder)
    # os.chdir(savefig_folder)
    # savefig_folder=os.getcwd()
    # # generate plots of msd's
    # savefig_fn = os.path.basename(output_file_name).replace('.csv','_long_time_std.png')
    # retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,4],ylim=[0,10],saving = True,fontsize =22,figsize=(9,6),D=3.5)
    #
    # savefig_fn = os.path.basename(output_file_name).replace('.csv','_short_time_std.png')
    # retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,0.2],ylim=[0,1],saving = True,fontsize =22,figsize=(9,6),D=3.5)
    #
    # savefig_fn = os.path.basename(output_file_name).replace('.csv','_very_short_time_std.png')
    # retval = PlotMSD(df, t_values, msd_values, std_values, savefig_folder,savefig_fn,xlim = [0,0.05],ylim=[0,0.2],saving = True,fontsize =22,figsize=(9,6),D=3.5)
    # return output_file_name
