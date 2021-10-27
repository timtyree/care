import dask.bag as db, time, os
import numpy as np, cudf, cupy as cp, dask_cudf
from ..utils.utils_cu import get_DT_cu
from .identify_all_coexisting_trajectory_pairs import identify_all_coexisting_pairs
from .extract_all_coexisting_trajectory_pairs import extract_trajectory_pairs_cu
from .distance_func import comp_xy_distance_L2_pbc_cu
from scipy.signal import savgol_filter
##########################################
# Helper Functions
##########################################
#TODO(later): implement savgol_filter_cu as gpu accelerated savistzky-golay filtration
def select_annihilating_pairs(df,
                              df_intersecting_pairs_all,
                              DS,
                              max_dtmax_thresh,
                              max_Rfinal_thresh,
                              min_duration_thresh,
                              width,
                              height,
                              trial_col='event_id_int',
                              pid_col='particle',
                              t_col='t',
                              **kwargs):
    #compute the xy_self and xy_other locations for each row in df_intersecting_pairs_all
    dff = df.set_index([trial_col, pid_col, t_col])
    df_pairs = df_intersecting_pairs_all.set_index(
        [trial_col, 'pid_self', 'tmax']).copy()
    df_pairs[['x_self', 'y_self']] = dff.loc[df_pairs.index][['x', 'y']]
    df_pairs = df_pairs.reset_index().set_index(
        [trial_col, 'pid_other', 'tmax'])
    df_pairs[['x_other', 'y_other']] = dff.loc[df_pairs.index][['x', 'y']]
    df_pairs = comp_xy_distance_L2_pbc_cu(df_pairs, width, height)
    df_pairs['Rfinal'] = df_pairs['dist'] * DS
    df_pairs = df_pairs.reset_index().set_index(
        [trial_col, 'pid_self', 'pid_other'])
    #compute the difference in time between the last observation of either particle
    df_pairs['dtmax'] = cp.abs(df_pairs['tmax_self'] - df_pairs['tmax_other'])
    # #DONE: visualize the surviving values and numbers for a sanity check
    # dtmax_values,Rfinal_values=df_pairs[['dtmax','Rfinal']].values.T.get()
    # # plt.scatter(dtmax_values,Rfinal_values,alpha=0.01)
    # # duration_values=df_pairs[(df_pairs['Rfinal']<=max_Rfinal_thresh)&(df_pairs['dtmax']<=max_dtmax_thresh)].duration.values.get()
    # # plt.hist(duration_values)
    # # print(np.min(duration_values))
    # # plt.show()

    bookeep = (df_pairs['Rfinal'] <= max_Rfinal_thresh) & (
        df_pairs['dtmax'] <= max_dtmax_thresh) & (df_pairs['duration'] >=
                                                  min_duration_thresh)
    df_pairs = df_pairs.loc[bookeep].copy()
    return df_pairs


def drop_any_duplicate_pairs(df_pairs, testing=True, trial_col='event_id_int', **kwargs):
    if testing:
        #TODO: identify any repeated particles that are repeated
        df_pairs = df_pairs.reset_index().set_index(
            [trial_col, 'pid_self', 'pid_other'])
        piv = df_pairs.index.values
        trial_lst = sorted(set(piv[:, 0].get()))
        for trial in trial_lst:
            iv = piv[piv[:, 0] == trial][:, 1:]
            #ensure that no particle appears twice in iv
            total_num_pair_members = iv.flatten().shape[0]
            unique_member_lst=list(set(iv.flatten().get()))
            no_duplicates_exist = total_num_pair_members == len(unique_member_lst)

            if not no_duplicates_exist:
                Rfinal_is_in_col_lst={'Rfinal'}.issubset(set(df_pairs.columns))
                if not Rfinal_is_in_col_lst:
                    raise("Warning: Rfinal missing from column list!")
                #build a new df_pairs from the ground up by selecting the minimum final range indicies
                dfp=df_pairs.reset_index()
                id_keep_lst=[]
                for member_id in unique_member_lst:
                    boo=(dfp['pid_self']==member_id)|(dfp['pid_other']==member_id)
                    id_keep=int(dfp[boo].sort_values(by='Rfinal').head(1).index.values.get())
                    id_keep_lst.append(id_keep)

                df_pairs=dfp.iloc[id_keep_lst].copy()

        #         #TODO: identify the duplicated particle
        #         #TODO: select all entries in df_pairs that involve the duplicated particle
        #         #TODO: select the particle pair that is nearest
        #         print(
        #             "Warning: duplicate particles detected but not removed!  TODO: Implement the 3 comments above!"
        #         )

        #     assert (no_duplicates_exist)
    return df_pairs



def comp_time_to_end_cu(dfr,df_pairs, trial_col='event_id_int', t_col='t', round_t_to_n_digits=7, **kwargs):
    event_col_lst=[trial_col,'pid_self','pid_other']
    #extract the xy values for observations in range if they are the final observation for a given pair
    grouped=dfr.reset_index().groupby(event_col_lst)
    max_times=grouped[t_col].max().reset_index().values
    last_obs=dfr.reset_index().set_index([trial_col,'pid_self','pid_other',t_col]).loc[max_times.get().T]
    last_range=last_obs['R'].values.T

    last_obs['tf']=last_obs.index.values[:,-1]
    last_times=last_obs.reset_index().set_index(event_col_lst)['tf']

    df_pairs=df_pairs.reset_index().set_index(event_col_lst)
    df_pairs['tf']=last_times

    #compute tmax for every row in dfr
    dfr=dfr.reset_index().sort_values(event_col_lst).set_index(event_col_lst)
    index_values=dfr.index.values.get().T
    # df_pairs.set_index(event_col_lst).loc[index_values.T,'tmax']
    # tmax_values=df_pairs.sort_values(event_col_lst).set_index(event_col_lst).loc[index_values,'tmax'].values
    tmax_values,num_rows_values=df_pairs.reset_index().sort_values(event_col_lst).set_index(event_col_lst)[['tf','num_rows']].values.T
    tf_values=cp.repeat(tmax_values.get(),num_rows_values.get().astype(int))
    dfr['tdeath']=cp.around(tf_values-dfr[t_col],round_t_to_n_digits)
    return dfr

def compute_radial_velocities_of_annihilations_cu(df,
                                                  navg2, #num. frames to average over.  navg2 must be an odd, positive integer.
                                                  DS, #cm/pixel
                                                  DT, #ms/frame
                                                  width, #pixels per x coord
                                                  height, #pixels per y coord
                                                  max_dtmax_thresh=0,
                                                  max_Rfinal_thresh=0.2, #cm
                                                  min_duration_thresh=40, #ms
                                                  use_tavg2=True,
                                                  trial_col='event_id_int',
                                                  pid_col='particle',
                                                  t_col='t',
                                                  round_t_to_n_digits=7,
                                                  printing=False,
                                                  testing=True,
                                                  **kwargs):
    '''df is a cudf.DataFrame instance
    Example Usage:
    navg2=int(tavg2/DT)
    if not (navg2%2==1): #odd navg2 is required by savgol_filter
        navg2+=1
        tavg2=np.around(navg2*DT,7)
    dfr,df_pairs=compute_radial_velocities_of_annihilations_cu(df,
                                                  navg2, #num. frames to average over.  navg2 must be an odd, positive integer.
                                                  DS, #cm/pixel
                                                  max_dtmax_thresh=0, #ms
                                                  max_Rfinal_thresh=0.2, #cm
                                                  min_duration_thresh=40, #ms
                                                  width=200,
                                                  height=200,
                                                  use_tavg2=True,
                                                  trial_col='event_id_int',
                                                  pid_col=pid_col,
                                                 printing=True, testing=True)
    '''
    df[t_col] = cp.around(df[t_col], round_t_to_n_digits)
    df = df.sort_values([trial_col, pid_col, t_col], ascending=True).copy()

    #compute the intermediate dataframe of particle start/end times
    grouped = df.groupby([trial_col, pid_col])
    dft = grouped[t_col]
    dfu = cudf.DataFrame({
        'tmin': dft.min(),
        'tmax': dft.max(),
    })
    dfu.reset_index(inplace=True)
    df_intersecting_pairs_all = identify_all_coexisting_pairs(
        df=dfu.copy(), pid_col=pid_col, t_col=t_col, trial_col=trial_col, printing=printing)

    df_pairs = select_annihilating_pairs(df, df_intersecting_pairs_all, DS,
                                         max_dtmax_thresh, max_Rfinal_thresh,
                                         min_duration_thresh,
                                         width,
                                         height,
                                         trial_col=trial_col,
                                         pid_col=pid_col,
                                         t_col=t_col)

    df_pairs = drop_any_duplicate_pairs(df_pairs, testing=testing)
    if printing:
        n_pairs_all = df_intersecting_pairs_all.shape[0]
        n_pairs_annihilating = df_pairs.shape[0]
        print(
            f"selected {n_pairs_annihilating} particle pairs as annihilating out of {n_pairs_all} possible particle pairs"
        )

    #extract trajectories for the selected pairs
    df_pairs.reset_index(inplace=True)
    df_traj = extract_trajectory_pairs_cu(df, df_pairs, pid_col, t_col,
                                          trial_col, DT)
    df_traj = comp_xy_distance_L2_pbc_cu(df_traj, width, height)
    df_traj['R'] = df_traj['dist'] * DS
    dfr = df_traj[[
        trial_col, 'pid_other', 'pid_self', t_col, 'R', 'index_pairs',
        'index_self', 'index_other', 'x_self', 'y_self', 'x_other', 'y_other'
    ]]

    dfr['R_nosavgol'] = dfr['R']
    if use_tavg2:
        grouped = dfr.to_pandas().groupby([trial_col, 'pid_self', 'pid_other'])
        #compute the savgol_filtered as R
        savgol0_kwargs = dict(window_length=navg2,
                              polyorder=3,
                              deriv=0,
                              delta=1.0,
                              axis=-1,
                              mode='interp')

        result = grouped['R'].apply(savgol_filter, **savgol0_kwargs)
        R_values = cp.array(np.concatenate(result.values))
        dfr['R'] = R_values



    #compute time until death
    dfr = comp_time_to_end_cu(dfr=dfr,
                              df_pairs=df_pairs,
                              trial_col=trial_col,
                              t_col=t_col,
                              round_t_to_n_digits=round_t_to_n_digits)
    return dfr,df_pairs

##########################################
# Main Routine
##########################################
def routine_compute_radial_velocities_pbc_cu(input_fn,
                                             tavg2,
                                             width,  #for handling periodic boundary conditions (pbc)
                                             height, #for handling periodic boundary conditions (pbc)
                                             ds,     #domain size in cm.  set to width to have nulled effect
                                             max_dtmax_thresh = 0,      #ms
                                             max_Rfinal_thresh = 0.2,   #cm
                                             min_duration_thresh = 40,  #ms
                                             round_t_to_n_digits = 7,   #digits
                                             trial_col='event_id_int',
                                             pid_col='particle',
                                             t_col='t',
                                             use_tavg2=True,
                                             save_df_pairs=True,
                                             printing=True,
                                             testing=True,
                                             **kwargs):
    '''loads an input_fn, uses compute_radial_velocities_of_annihilations_cu, and saves as ...+'_annihilations_denoised.csv' with index=False
    Example Usage:
    input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-3-LR/param_qu_tmax_30_Ko_5.4_diffCoef_0.0005_dt_0.5/smoothed_trajectories_tavg_4/ic002.11_traj_sr_600_mem_0_smoothed.csv"
    dfr_dir=routine_compute_radial_velocities_pbc_cu(input_fn,
                                                 tavg2=24,
                                                 width=200,  #for handling periodic boundary conditions (pbc)
                                                 height=200, #for handling periodic boundary conditions (pbc)
                                                 ds=5,     #domain size in cm.  set to width to have nulled effect
                                                 max_dtmax_thresh = 0,      #ms
                                                 max_Rfinal_thresh = 0.2,   #cm
                                                 min_duration_thresh = 40,  #ms
                                                 round_t_to_n_digits=7,
                                                 trial_col='event_id_int',
                                                 pid_col='particle',
                                                 t_col='t',
                                                 use_tavg2=True,
                                                 save_df_pairs=True,
                                                 printing=True,
                                                 testing=True)#,**kwargs)
    '''
    #TODO(later): enable multiple trials to be processed by the same call to using dask_cudf
    # df=dask_cudf.read_csv(input_fn_lst).compute()
    df = cudf.read_csv(input_fn)
    DT = get_DT_cu(df, t_col, pid_col, round_digits=round_t_to_n_digits)

    #compute the trial column if it is not already present
    has_trial_col={trial_col}.issubset(ss)
    if not has_trial_col:
        compute_event_id(df,input_fn,pid_col=pid_col)
        trial_col='event_id_int'
        if printing:
            print(f"trial_col missing... resetting trial_col to default value {trial_col}")


    # DT=np.around(get_DT(df, t_col=t_col, pid_col=pid_col),5)
    DS = ds / width  #cm per pixel lengthscale
    if printing:
        print(f"DT={DT} ms")
    navg2 = int(tavg2 / DT)
    if not (navg2 % 2 == 1):  #odd navg2 is required by savgol_filter
        navg2 += 1
        tavg2 = np.around(navg2 * DT, round_t_to_n_digits)
    if use_tavg2:
        if printing:
            print(f'savistzky-golay time window in use: {tavg2} ms')
        if testing:
            assert (min_duration_thresh > tavg2)

    #main routine
    dfr, df_pairs = compute_radial_velocities_of_annihilations_cu(
        df,
        navg2,  #num. frames to average over.  navg2 must be an odd, positive integer.
        DS,     #cm/pixel
        DT,     #ms/frame
        width,
        height,
        max_dtmax_thresh=max_dtmax_thresh,  #ms
        max_Rfinal_thresh=max_Rfinal_thresh,  #cm
        min_duration_thresh=min_duration_thresh,  #ms
        use_tavg2=use_tavg2,
        pid_col=pid_col,
        printing=printing,
        testing=testing)


    #infer filesystem for saving functionally from inputs
    trgt='_tavg'
    tavg1=float(eval(input_fn[input_fn.find(trgt)+len(trgt):].split('/')[0].split('_')[-1]))
    ext = f'_annihilations_denoised.csv'
    folder_ext=f'_tavg1_{tavg1}_tavg2_{tavg2}_maxdtmax_{max_dtmax_thresh}_maxRfin_{max_Rfinal_thresh}_mindur_{min_duration_thresh}'
    save_fn = os.path.basename(input_fn).replace('.csv', ext)

    if save_df_pairs:
        #save df_pairs data from input_fn in dfr
        save_folder = os.path.dirname(
            os.path.dirname(input_fn)
        ) + f'/pairs'+folder_ext
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        os.chdir(save_folder)
        df_pairs.reset_index().to_csv(save_fn,index=False)
        # pairs_dir = os.path.join(save_folder, save_fn)
        pairs_dir = os.path.abspath(save_fn)

    #save dfr data from input_fn in dfr
    save_folder = os.path.dirname(
        os.path.dirname(input_fn)
    ) + f'/smoothed_annihilations'+folder_ext
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    os.chdir(save_folder)
    dfr.reset_index().to_csv(save_fn,index=False)
    # dfr_dir = os.path.join(save_folder, save_fn)
    dfr_dir = os.path.abspath(save_fn)

    if printing:
        print(f"radial velocities of annihilation events were successfully saved to csv")
    return dfr_dir
