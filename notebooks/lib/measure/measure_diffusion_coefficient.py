import pandas as pd, numpy as np, trackpy
from scipy import stats
from .compute_slope import *

def compute_emsd(traj,DT,omit_time=0,printing=False,DS=0.025):
    '''traj is a pandas.DataFrame instance with columns "frame", "particle", "x" and "y" containing the results of unwrapped trajectories.
    returns a pandas.DataFrane instance containing the ensemble mean squared displacement.'''
    #remove the first and last 150ms of each input trajectory by default
    n_rows_omit=int(omit_time/DT)
    indices_of_head=traj.groupby('particle').head(n_rows_omit).index
    indices_of_tail=traj.groupby('particle').tail(n_rows_omit).index
    drop_id_lst=list(indices_of_head.values)
    drop_id_lst.extend(list(indices_of_tail.values))
    traj.drop(index=drop_id_lst,inplace=True)
    #truncate max lag to the min lifetime
    len_lst=traj.groupby('particle').x.count().values
    if len(len_lst)==0:
        return None
    max_lag=np.min(len_lst)#frames
    #compute ensemble MSD
    emsd=trackpy.emsd(
        traj=traj,
        mpp=DS,
        fps=1/DT,
        max_lagtime=max_lag,
        detail=False,
        pos_columns=None,
    )
    if printing:
        print(f'max_lag was {DT*max_lag} ms')
    return emsd


def compute_Dhat(emsd,tau_min=300,window_width=50):
    '''
    Dhat,Delta_Dhat,tau_min,tau_max,Rsquared,delta_tau=compute_Dhat(emsd,tau_min=300,window_width=50)
    '''
    #TODO: compute D resulting from slope of MSD to D resulting from intercept of log-log plot of MSD
    tau_max=tau_min+window_width
    msd_values=emsd.values
    lag_values=emsd.index.values
    boo=(lag_values>=tau_min)&(lag_values<=tau_max)
    x = lag_values[boo]/10**3
    y = msd_values[boo]
    # D resulting from slope of MSD
    dict_output=compute_95CI_ols(x=x,y=y)
    # print(f"By slope of MSD, D = {dict_output['m']/4} +- {dict_output['Delta_m']/4} cm^2/s")
    Dhat=dict_output['m']/4
    Delta_Dhat=dict_output['Delta_m']/4
    #     print(f"By slope of MSD, D = {Dhat} +- {Delta_Dhat=} cm^2/s")
    Rsquared=dict_output['Rsquared']
    #     D_expval=slope
    #     D_stderr=std_err
    delta_tau=window_width
    #     Rsquared=r_value**2
    return Dhat,Delta_Dhat,tau_min,tau_max,Rsquared,delta_tau

def compute_v_rms(emsd,num_points=8):
    '''returns root mean square velocity from the slope of the first num_points points listed in emsd.
    emsd is the ensemble mean squared displacement in units of squared pixels'''
    NV=num_points
    lag_values=emsd.index.values[:NV]/10**3
    rmsd_values=np.sqrt(emsd.values)[:NV]
    dict_output=compute_95CI_ols(x=lag_values,y=rmsd_values)
    v_rms=dict_output['m']
    Delta_v_rms=dict_output['Delta_m']
    #     print(f"By slope of RMSD, v_rms = {v_rms} +- {Delta_v_rms} cm/s")
    Rsquared=dict_output['Rsquared']
    window_width=lag_values[-1]*10**3
    return v_rms,Delta_v_rms,0.,window_width,Rsquared,window_width


def compute_Dbar(emsd,MSD_thresh=1.,max_lagtime=500):
    '''returns D_expval,D_stderr,tau_min,tau_max,Rsquared,delta_tau
    emsd is the ensemble mean squared displacement in units of squared pixels,
    indexed by lag in units of frames.
    DT is the time between two frames (ms)
    DS is the distance between two pixels (cm).
    D_expval is in units of cm^2/ms.'''
    # if emsd is None:
    #     return None
    x_values=emsd.index.values#*DT
    y_values=emsd.values#*DS**2

    #choose tau_max
    boo_meanders=y_values-MSD_thresh>0
    if not boo_meanders.any():
        return None
    tau_min=x_values[np.argwhere(boo_meanders)[0,0]]
    tau_max=tau_min+max_lagtime
    last_tau=x_values[-1]
    if tau_max>last_tau:
        tau_max=last_tau

    #restrict observation to tau_min to tau_max
    boo=(x_values>=tau_min)&(x_values<=tau_max)
    #compute slope with scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values[boo],y_values[boo])
    #    std_err: D_stderr : Standard error of the estimated gradient.
    #    slope  : D_expval : diffcoef of msd curve
    D_expval=slope
    D_stderr=std_err
    delta_tau=tau_max-tau_min
    Rsquared=r_value**2
    return D_expval,D_stderr,tau_min,tau_max,Rsquared,delta_tau

def compute_time_between_frames(df,round_t_to_n_digits=3):
    #get the time between two frames
    first_frames=df.frame.values[:2]
    DT =df[df.frame==first_frames[1]].t.values[0]
    DT-=df[df.frame==first_frames[0]].t.values[0]#ms
    return np.around(DT,round_t_to_n_digits)

if __name__=='__main__':
    import sys,os
    for file in sys.argv[1:]:
        # trgt  ='_unwrap.csv'
        # assert (file[-len(trgt):]==trgt)
        traj  =pd.read_csv(file)
        DT    =compute_time_between_frames(df=traj)
        emsd  =compute_emsd(traj,DT,omit_time=0,printing=False)
        retval= compute_Dbar(emsd,DT,DS=0.025,MSD_thresh=1.,max_lagtime=500)
        D_expval,D_stderr,tau_min,tau_max,Rsquared,delta_tau=retval
        print(f'printing results for {os.path.basename(file)}')
        print(retval)
