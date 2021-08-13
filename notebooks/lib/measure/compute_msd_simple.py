import numpy as np, pandas as pd,os
from ..utils.utils_traj import unwrap_traj_and_center
########################################################
# Compute Mean Squared Displacements via particle averaging
########################################################
def return_msd_particle_average(input_fn,ds,width,height,use_unwrap,DT,pid_col='pid_explicit',t_col='t',
                                **kwargs):
    '''
    input_fn is a .csv locating a trajectory file with particles identified by pid_col
    and time indicated by t_col
    ds is the total domain size and width and height are the number of length units / pixels afforded to the original computational domain.
    kwargs are passed to unwrap_traj_and_center

    previously named return_msd_phys
    TODO: GPU accelerate this pandas-like function with rapids cudf
    '''
    df=pd.read_csv(input_fn)
    trial_folder_name=os.path.dirname(os.path.dirname(input_fn))


    #(bad)particle data analyzed using full model pipeline
    # input_fn="/Users/timothytyree/Documents/GitHub/bgmc/python/data/local_results/euic_False_fc_2_r_0.1_D_2_L_10_kappa_1500_varkappa_0/trajectories_unwrap/pbc_particle_log121_traj_sr_30_mem_0_unwrap.csv"
    # width=10 #width of computational domain
    # ds   =10  #cm
    # #from here on, we will use units in terms of those used by the full model
    # height=width
    DS=ds/width
    if use_unwrap is True:
        #unwrap trajectories
        pid_lst = sorted(set(df[pid_col].values))
        #(duplicates filtered earlier in full model pipeline.  Unnecessary in particle model with explicit tracking_ _  _ _ ) filter_duplicate_trajectory_indices is slow (and can probs be accelerated with a sexy pandas one liner)
        # pid_lst_filtered = filter_duplicate_trajectory_indices(pid_lst,df)
        df = pd.concat([unwrap_traj_and_center(df[df[pid_col]==pid], width=width, height=height, **kwargs) for pid in pid_lst])
    # DT=get_DT(df,pid_col=pid_col) #ms
    # df[df.frame==2].describe()
    df['sd']=df['x']**2+df['y']**2
    d_msd=df.groupby('t')['sd'].mean()
    lagt_values=d_msd.index.values
    msd_values=d_msd.values*DS**2
    return lagt_values,msd_values

########################################################
# Compute Mean Squared Displacements via time averaging
########################################################
def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(r):
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2

def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds

if __name__=='__main__':
    #test that msd_fft results in msd_straight_forward to machine precision for an unseeded random walk
    N=1000
    r = np.cumsum(np.random.choice([-1., 0., 1.], size=(N, 3)), axis=0)
    assert ( np.isclose(msd_fft(r)-msd_straight_forward(r),0.).all() )
    numerical_tolerance=np.mean(msd_fft(r)-msd_straight_forward(r))
    print(f"the numerical aggrement between slower and faster methods was numerical_tolerance = {numerical_tolerance}.")
