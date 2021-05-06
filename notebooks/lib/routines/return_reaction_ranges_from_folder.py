import numpy as np,pandas as pd, matplotlib.pyplot as plt, dask.bag as db
from .compute_interactions import compute_df_interactions
# TODO: get bins
# TODO: wrap this in a function that takes a .csv trajectory input_fn, and a set of bins and returns counts in those bins
# TODO: run on a folder of (still wrapped) trajectories

def compute_DT(df,round_t_to_n_digits=3):
    '''DT is the time between two observations'''
    DT=np.around(df[(df.frame==1)].t.values[0]-df[(df.frame==0)].t.values[0],round_t_to_n_digits)
    return DT

def return_bd_ranges(input_fn,DS=5./200.,round_t_to_n_digits=3):
    df=pd.read_csv(input_fn)
    DT=compute_DT(df,round_t_to_n_digits=round_t_to_n_digits)
    #compute interactions
    df_interactions=compute_df_interactions(input_fn,DS=DS)
    df_interactions.dropna(inplace=True)
    death_ranges=DS*df_interactions.rT.values
    birth_ranges=DS*df_interactions.r0.values
    return death_ranges,birth_ranges,DT

def get_bin_edges(input_fn,ds=5.,width=200.,nbins=40):
    death_ranges,birth_ranges,DT=return_bd_ranges(input_fn,DS=ds/width,round_t_to_n_digits=3)
    a=death_ranges
    bin_totals,bin_edges=np.histogram(
        a,bins=nbins)#,range=None,density=None)
    bin_edges-=bin_edges[0] #tare the origin
    return bin_edges

def comp_bdrange_bincounts(input_fn,bin_edges,ds,width):
    death_ranges,birth_ranges,DT=return_bd_ranges(input_fn,DS=ds/width,round_t_to_n_digits=3)
    # returned index i of bin_totals satisfies
    # bin_edges[i-1] <= x < bin_edges[i] by default
    #first bin width
    bin_width=np.diff(bin_edges)[0]
    # # assert equal bin width (not needed)
    # assert((np.diff(bin_edges)==bin_width).all())
    bin_count_death,_=np.histogram(death_ranges,bins=bin_edges)
    bin_count_birth,_=np.histogram(birth_ranges,bins=bin_edges)
    return bin_count_death, bin_count_birth,DT

def sum_bin_count(bin_count1,bin_count2):
    '''supposes same binning'''
    return bin_count1+bin_count2

def comp_bdrates_by_bin(bin_count_birth,bin_count_death,bin_edges,DT):
    net_bin_count_death = np.sum(bin_count_death)
    net_bin_count_birth = np.sum(bin_count_birth)
    DT_sec=DT/10**3 # seconds between two observations
    bin_ranges=(bin_edges[1:]+bin_edges[:-1])/2.
    bin_widths=(bin_edges[1:]-bin_edges[:-1])
    drate_values=bin_count_death/bin_widths/net_bin_count_death/DT_sec
    brate_values=bin_count_birth/bin_widths/net_bin_count_birth/DT_sec
    range_values=bin_ranges
    return range_values,brate_values,drate_values

def PlotRangesBD(bin_ranges,brate_values,drate_values,DT,fontsize=16,figsize=(7,5)):
    '''plots results of what was aggregated from a large number of tips'''
    bin_width=np.diff(bin_ranges)[0]
    fig,ax=plt.subplots(figsize=figsize)
    ax.bar(x=bin_ranges,height=drate_values,align='center',width=bin_width,alpha=0.7,color='red',label='pair annihilation rate')
    ax.bar(x=bin_ranges,height=brate_values,align='center',width=bin_width,alpha=0.7,color='green',label='pair creation rate')
    ax.set_xlabel('apparent range (cm)',fontsize=fontsize)
    ax.set_ylabel('reaction rate (Hz)',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    DT_sec=DT/10**3
    ax.set_title(f"{DT_sec:.4f} seconds between two frames",fontsize=fontsize)
    return fig,ax

if __name__=="__main__":
    input_fn=f"{nb_dir}/Data/initial-conditions-suite-2/ds_5_param_set_8_V_0.5_fastkernel/ic_200x200.001.13_traj_sr_400_mem_0.csv"
    bin_edges=get_bin_edges(input_fn,ds=5.,width=200.,nbins=40)

    #TODO: use dask.bag on UB
    input_fn_lst=[input_fn]
    for input_fn in input_fn_lst:
        bin_count_death, bin_count_birth,DT=comp_bdrange_bincounts(input_fn,bin_edges,ds=5.,width=200.)

    #TODO: reduce using dask.bag
    # bin_count_death=sum_bin_count(bin_count_death1,bin_count_death2)
    # bin_count_birth=sum_bin_count(bin_count_birth1,bin_count_birth2)

    range_values,brate_values,drate_values=comp_bdrates_by_bin(bin_count_birth,bin_count_death,bin_edges,DT)
    # #TODO: put ^this in a routine that asks for a trajectory folder as input
    #TODO: compute/print summary statistics
    # print(f"for this trial,")
    # print(f"\tdeath range was {np.mean(death_ranges):.3f} +- {np.std(death_ranges):.3f} cm")
    # print(f"\tbirth range was {np.mean(birth_ranges):.3f} +- {np.std(birth_ranges):.3f} cm")
    # print(f"\ttime between two frames was {DT} ms")

    # #Plot the results
    # fig,ax=PlotRangesBD(range_values,brate_values,drate_values,DT,fontsize=16,figsize=(7,5))
    # fig.show()
