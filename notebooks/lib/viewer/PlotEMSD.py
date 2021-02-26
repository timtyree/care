import matplotlib.pyplot as plt, numpy as np, pandas as pd
from ..utils import *
from ..measure import *

def plot_emsd(ax,emsd,label='_Hidden', color='gray', alpha=1.,**kwargs):
    x_values=emsd.index.values/10**3#np.log10(emsd.index.values)# lag in seconds
    y_values=emsd.values#*10**3#np.log10(emsd.values) #msd in cm^2
    ax.plot(x_values,y_values, label=label, color=color, alpha=alpha,**kwargs)

def format_plot_emsd(ax,fontsize=20,use_loglog=True):
    #format plot
    if use_loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('lag (seconds)',fontsize=fontsize)
    ax.set_ylabel('MSD (cm$^2$)',fontsize=fontsize)
#     ax.set_title(f'FK model, Area:25cm$^2$, $D_{{V_{{mem}}}}$:0.5cm$^2$/s, N:{trials_considered}\nmin_duration:{T_min/10**3:.1f}s\n',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
#     ax.set_ylim([0,2.05])

# def plot_slope_of_emsd(ax,emsd,T_min,omit_time,label='_Hidden', color='gray', alpha=0.3,plot_reference_lines=True,**kwargs):
#     lag_values,slope_values=compute_slope_vs_lag(emsd,T_min,omit_time,window_width=50,stepsize=10)
def plot_slope_of_emsd(ax,lag_values,slope_values,label='_Hidden', color='gray', alpha=1.,plot_reference_lines=True,**kwargs):
#     lag_values,slope_values=compute_slope_vs_lag(emsd,T_min,omit_time,window_width=50,stepsize=10
    ax.plot(lag_values,slope_values, label=label, color=color, alpha=alpha,**kwargs)
    if plot_reference_lines:
        ax.plot(lag_values,2+0.*slope_values,label='ballistic')
        ax.plot(lag_values,1+0.*slope_values,label='diffusive')
def format_slope_of_emsd(ax,fontsize=20,use_loglog=True):
    #format plot
    if use_loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('lag (seconds)',fontsize=fontsize)
    ax.set_ylabel('exponent value',fontsize=fontsize)
#     ax.set_title(f'FK model, Area:25cm$^2$, $D_{{V_{{mem}}}}$:0.5cm$^2$/s, N:{trials_considered}\nmin_duration:{T_min/10**3:.1f}s\n',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    ax.legend(loc='best',fontsize=fontsize)
#     ax.set_ylim([0,2.05])


if __name__=='__main__':
    import sys,os
    for file in sys.argv[1:]:
        # trgt  ='_unwrap.csv'
        # assert (file[-len(trgt):]==trgt)
        df  =pd.read_csv(file)
        T_min=1000#ms
        omit_time=150#ms
        DS=0.025#cm/pixel
        figsize=(17,4);fontsize=16
        saving=True
        savefig_folder=os.path.dirname(file)#os.path.join(nb_dir,'Figures/msd_loglog')
        savefig_fn=os.path.basename(file).replace('.csv','.png')#f'logMSD_vs_loglag_Tmin_{T_min/10**3:.1f}_N_{trials_considered}_mni_{min_num_individuals}.png'

        df=pd.read_csv(file)
        # df=return_unwrapped_trajectory(df, width, height, sr, mem, dsdpixel, **kwargs)
        DT=compute_time_between_frames(df);print(f"DT={DT}")
        # df=get_all_longer_than(df,DT,T_min=T_min)
        #count remaining individuals
        num_individuals=len(list(set(df.particle.values)));print(f"num_individuals={num_individuals}")
        emsd=compute_emsd(traj=df.copy(), DT=DT, omit_time=omit_time, printing=False,DS=DS)

        fig,axs=plt.subplots(ncols=3,figsize=figsize)
        plot_emsd(axs[0],emsd)
        format_plot_emsd(axs[0],use_loglog=False,fontsize=fontsize)

        plot_emsd(axs[1],emsd)
        format_plot_emsd(axs[1],use_loglog=True,fontsize=fontsize)

        plot_slope_of_emsd(axs[2],emsd,label='_Hidden', color='gray', alpha=0.3,plot_reference_lines=True)
        format_slope_of_emsd(axs[2],fontsize=fontsize,use_loglog=False)

        if not saving:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            os.chdir(savefig_folder)
            plt.savefig(savefig_fn, dpi=300)
            print(f"saved figure in \n\t{savefig_fn}")
            plt.close()
