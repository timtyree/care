# PlotMSD.py
# Tim Tyree
# 12.18.2020
from ..my_initialization import *

def compute_emsd_for_longest_trajectories(input_file_name,n_tips = 1,DS = 5/200,DT = 1., round_t_to_n_digits=0, mint=100):
    #select the longest n trajectories
    df = pd.read_csv(input_file_name)
    df = df[df.t>mint].copy() #chop off times before mint
    df.reset_index(inplace=True)
    s = df.groupby('particle').t.count()
    s = s.sort_values(ascending=False)
    pid_longest_lst = list(s.index.values[:n_tips])
    #     df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])

    #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    df_lst = []
    for pid in  pid_longest_lst:#[2:]:
        d = df[(df.particle==pid)].copy()
        x_values, y_values = d[['x','y']].values.T
        index_values = d.index.values.T
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS=5/200,DT=1, jump_thresh=10.)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)

    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
#     assert ( (np.array(sorted(set(df_traj['particle'].values)))==np.array(sorted(pid_longest_lst))).all())
    #compute ensemble mean squared displacement
    emsd = trackpy.motion.emsd(df_traj, mpp=1., fps=1.,max_lagtime=40000)
    #cast ensemble mean squared displacement into units of cm^2 and seconds
    return pd.DataFrame({'msd':DS**2*emsd.values, 'lagt':emsd.index.values*DT/10**3, 'src':input_file_name})


def compute_average_msd(df, DT=1.):
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

def PlotMSD(df, t_values, msd_values, savefig_folder,savefig_fn,xlim = [0,0.05],ylim=[0,4],D = 75,saving = True,fontsize =22,figsize=(9,6)):
    '''plot msd for each trial listed in df.src. also plot average msd.'''
    #compute average msd by trial for a subset of trials
    fig, ax = plt.subplots(figsize=figsize)
    # x_values = ef.index.values#*DT/10**3 #lag in seconds
    # y_values = ef.values#*DS**2 #msd

    for src in src_lst:
        x_values = df[df.src==src].lagt.values
        y_values = df[df.src==src].msd.values
        ax.plot(x_values,y_values,c='blue',alpha=0.2)
    ax.plot(t_values,msd_values,c='g',lw=2)
    ax.plot(t_values,t_values*D,c='r',lw=2)

    # DS = 5/200 #cm per pixel
    # DT = 1. #ms per frame
    ax.set_xlim(xlim)
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
        print(f"saved figure in \n\t{savefig_fn}")
        plt.close()
    return {'t':t_values,'msd':msd_values}

def generate_msd_figures_routine(file,n_tips):
    V_thresh = eval(file[file.find('_V_')+len('_V_'):].split('_')[0])
    # get all .csv files in the working directory of ^that file
    folder_name = os.path.dirname(file)
    os.chdir(folder_name)
    retval = os.listdir()
    file_name_list = list(retval)
    # check each file if it ends in .csv before merging it
    def is_csv(file_name):
        return file_name[-4:]=='.csv'
    file_name_list = [f for f in file_name_list if is_csv(f)]
    # remove all files with 'threshold'
    # file_name_list = [f for f in file_name_list if f.find('threshold')==-1]

    print(f"Num. file names in list = {len(file_name_list)}.")
    #compute ensemble mean squared displacement for the longest n_tips for each trial in file_name_list
    os.chdir(folder_name)
    df = pd.concat([compute_emsd_for_longest_trajectories(input_file_name, n_tips=n_tips) for input_file_name in file_name_list])
    df.reset_index(inplace=True,drop=True)
    #save results
    dirname = os.path.dirname(file).split('/')[-1]
    save_folder = folder_name.replace(dirname,'msd')
    os.chdir(save_folder)
    output_file_name = f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv"
    df.to_csv(output_file_name, index=False)

    #compute average msd by trial for a subset of trials
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

    t_values, msd_values = compute_average_msd(df, DT=1.)
    savefig_folder = os.path.join(nb_dir,f'Figures/msd/V_{V_thresh}')

    # generate plots of msd's
    savefig_fn = os.path.basename(output_file_name).replace('.csv','_long_time.png')
    retval = PlotMSD(df, t_values, msd_values, savefig_folder,savefig_fn,xlim = [0,4],ylim=[0,10],saving = True,fontsize =22,figsize=(9,6),D=3.5)

    savefig_fn = os.path.basename(output_file_name).replace('.csv','_short_time.png')
    retval = PlotMSD(df, t_values, msd_values, savefig_folder,savefig_fn,xlim = [0,0.2],ylim=[0,1],saving = True,fontsize =22,figsize=(9,6),D=3.5)

    savefig_fn = os.path.basename(output_file_name).replace('.csv','_very_short_time.png')
    retval = PlotMSD(df, t_values, msd_values, savefig_folder,savefig_fn,xlim = [0,0.05],ylim=[0,0.2],saving = True,fontsize =22,figsize=(9,6),D=3.5)
    return ff, retval

# beep(4)
