# compute_diffcoeff.py
from ..my_initialization import *
from scipy import stats

def compute_diffusion_coeffs(input_file_name,tau_min=0.15,tau_max=0.5,**kwargs):
    '''consider input_file_name = emsd_longest_by_trial_tips_ntips_1_Tmin_0.15_Tmax_0.5.csv
    Example Usage:
        retval= compute_diffusion_coeffs_for_msd(input_file_name,tau_min=.15,tau_max=0.5)'''
    #select ranges that look linear are tau_min=.15 and tau_max=0.5

    #load csv msd data
    df=pd.read_csv(input_file_name)

    #initialize output dataframe by counting the number of trials that apppear in each tip log, listed in src
    ef=df.groupby('src').src.count()
    sv=ef.index;nv=ef.values
    df2=pd.DataFrame({'src':sv,'N':nv})

    # select msd data for single trial
    src_lst=list(set(df.src.values))

    for src in src_lst:
        # src=src_lst[0]
        tau_values,msd_values=df[df.src==src][['lagt','msd']].values.T

        boo=(tau_values>=tau_min)&(tau_values<=tau_max)
        x_values=tau_values[boo]
        y_values=msd_values[boo]
        duration_of_traj=tau_values[-1]

        if x_values.shape[0]>0:
            #compute diffusion coefficient
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values,y_values)
            #    std_err: D_stderr : Standard error of the estimated gradient.
            #    slope  : D_expval : diffcoef of msd curve
            D_expval=slope
            D_stderr=std_err
            #     retval=src,duration_of_traj,D_expval,D_stderr,intercept,r_value,p_value
        else:
            #no value was found
            D_expval=-9999.;D_stderr=-9999.;intercept=-9999.;r_value=-9999.;p_value=-9999.

        #update output dataframe with diffusion coefficient measurements
        df2.loc[df2.src==src,'D_expval']=D_expval#df2.loc[df2.src==src,
        df2.loc[df2.src==src,'D_stderr']=D_stderr#df2.loc[df2.src==src,
        df2.loc[df2.src==src,'duration_of_traj']=duration_of_traj#df2.loc[df2.src==src,
        df2.loc[df2.src==src,'intercept']=intercept#df2.loc[df2.src==src,
        df2.loc[df2.src==src,'Rsquared']=r_value**2#df2.loc[df2.src==src,
        df2.loc[df2.src==src,'p_value']=p_value#df2.loc[df2.src==src,


    #save results to csv
    os.chdir(os.path.dirname(input_file_name))
    savefn=("diffcoeff_"+os.path.basename(input_file_name)).replace('.csv',f'_Tmin_{tau_min}_Tmax_{tau_max}.csv')
    df2.to_csv(savefn)
    return savefn

def generate_diffcoeff_figures(input_file_name,trial_folder_name, dir_out=None, tau_min=.15,tau_max=0.5,saving=True,
        R2_thresh=0.,duration_thresh=0.,fontsize=22,figsize_2=(15,4.5),**kwargs):
    '''input_file_name is of the form:
        emsd_longest_by_trial_tips_ntips_1.csv'''
    if dir_out is None:
        #compute the diffusion coefficient summary
        os.chdir(os.path.dirname(input_file_name))
        dir_out="diffcoeff_summary_"+os.path.basename(input_file_name)


    # diffcoeff_emsd_longest_by_trial_tips_ntips_1_Tmin_0.15_Tmax_0.5.csv'''
    df=pd.read_csv(os.path.abspath(input_file_name),index_col=0)
    df=df[~df.D_expval.isnull()].copy()#drop trials that gave no tips long enough
    num_trials_total=df.N.values.shape[0]
    print(f"generating diffcoeff_figures for {os.path.basename(trial_folder_name)}")
    #filter results by whether a D_expval was found
    # print(f"""num. trials that didn't show a tip lasing longer than 150ms is {df[df.D_expval<-1000].N.size}.""")
    df=df[df.D_expval>=-1000].copy()

    #filter results by whether a D_expval was found
    # print(f"""num. trials that didn't show am Rsquared of at least {R2_thresh} is {df[df.D_expval<R2_thresh].N.size}.""")
    df=df[df.Rsquared>=R2_thresh].copy()
    print(f"""\tnum. trials that didn't show a tip lasing longer than {tau_min*10**3:.0f}ms is {df[df.D_expval<-1000].N.size}, and that didn't show am Rsquared of at least {R2_thresh} is {df[df.D_expval<R2_thresh].N.size}""")


    #plot results
    x_values= df.duration_of_traj.values
    y_values= df.D_expval.values
    yerr_values= df.D_stderr.values

    # nb_dir='/home/timothytyree/Documents/GitHub/care/notebooks/'
    here_dir=os.path.dirname(os.path.abspath(input_file_name))
    os.chdir(here_dir)
    savefig_folder = 'fig'#os.path.join(here_dir,'fig')#f'Figures/msd/'+trial_folder_name)
    if not os.path.exists(savefig_folder):
        os.mkdir(savefig_folder)
    os.chdir(savefig_folder)

    #generate plot 1
    fig,axs=plt.subplots(2)
    ax=axs[0]
    ax.scatter(x_values,y_values)
    ax.set_xlabel('duration of trajectory (sec)',fontsize=fontsize)
    ax.set_ylabel(r'$D$ (cm$^2$/s)',fontsize=fontsize)
    ax=axs[1]
    ax.scatter(x_values,yerr_values)
    ax.set_xlabel('duration of trajectory (sec)',fontsize=fontsize)
    ax.set_ylabel(r'$\Delta D$ (cm$^2$/s)',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    if not saving:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        # os.chdir(savefig_folder)
        savefig_fn=os.path.basename(input_file_name).replace('.csv','_plot_1.png')
        plt.savefig(savefig_fn, dpi=300)
        # print(f"saved figure in \n\t{savefig_fn}")
        plt.close()


    # print(f"it appears that the stderr of D is large for tips lasting less than 2.5 seconds")
    #filter trajectories shorter than 2.5 seconds in duration
    # print(f"""filtering {} trajectories shorter than {duration_thresh:.1f} seconds in duration""")
    df=df[df.duration_of_traj>=duration_thresh].copy()
    x_values= df.duration_of_traj.values
    y_values= df.D_expval.values
    yerr_values= df.D_stderr.values

    #generate plot 1
    fig,axs=plt.subplots(ncols=4,figsize=figsize_2)
    ax=axs[0]
    ax.scatter(yerr_values,df.Rsquared.values, alpha=0.5)
    ax.set_xlabel(r'$\Delta D$ (cm$^2$/s)',fontsize=fontsize)
    ax.set_ylabel(r'$R^2$',fontsize=fontsize)
    ax.set_ylim([R2_thresh,1])
    # ax.set_xlim([0,0.05])

    ax=axs[1]
    ax.scatter(yerr_values,y_values, alpha=0.5)
    ax.set_xlabel(r'$\Delta D$ (cm$^2$/s)',fontsize=fontsize)
    ax.set_ylabel(r'$D$ (cm$^2$/s)',fontsize=fontsize)
    # ax.set_xlim([0,0.05])

    ax=axs[2]
    ax.hist(y_values,bins=15, alpha=0.5)
    ax.set_ylabel(r'frequency',fontsize=fontsize)
    ax.set_xlabel(r'$D$ (cm$^2$/s)',fontsize=fontsize)

    ax=axs[3]
    ax.scatter(df.intercept.values,y_values, alpha=0.5)
    ax.set_xlabel(r'MSD intercept (cm$^2$)',fontsize=fontsize)
    ax.set_ylabel(r'$D$ (cm$^2$/s)',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    if not saving:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        # os.chdir(savefig_folder)
        savefig_fn=os.path.basename(input_file_name).replace('.csv','_plot_2.png')
        plt.savefig(savefig_fn, dpi=300)
        # print(f"saved two figures in {savefig_folder}")
        plt.close()

    ######################
    # Compute summary row
    ######################
    df[df.D_expval<-1000].N.size
    num_trials_considered=df.N.values.shape[0]
    # print(f"number of trials considered = {num_trials_considered}.")
    c=df.describe().T[['mean','std']].T

    # input_file_name='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ds_5_param_set_8_fastkernel_V_0.4_archive/msd/emsd_longest_by_trial_tips_ntips_1.csv'
    sll=os.path.basename(input_file_name)
    trial_name=trial_folder_name
    n_tips=eval(sll[sll.find('ntips_')+len('ntips_'):].split('_')[0])
    mean_D, stdev_D=c[['D_expval']].values
    mean_D=float(mean_D);stdev_D=float(stdev_D)
    mean_stderr_D=float(c[['D_stderr']].values[0])
    stdev_stderr_D=float(c[['D_stderr']].values[1])
    # num_trials_considered=df.N.values.shape[0]

    #BLUF
    df_out=pd.DataFrame({'trial_folder_name':[trial_name], 'n_tips':[n_tips],
                         "mean_D":[mean_D], "stdev_D":[stdev_D],
                         "mean_stderr_D":[mean_stderr_D], "stdev_stderr_D":[stdev_stderr_D],
                        "num_trials_considered":[num_trials_considered],
                        "num_trials_computed":[num_trials_total],
                         "tau_min":[tau_min],"tau_max":[tau_max],
                         "R2_thresh":[R2_thresh], "duration_thresh":[duration_thresh]})

    #save results to csv
    os.chdir(os.path.dirname(input_file_name))
    if dir_out is None:
        dir_out=os.path.abspath("diffcoeff_summary_"+os.path.basename(input_file_name))
    df_out.to_csv(dir_out)
    print(f"saved diffcoeff_summary to {os.path.dirname(dir_out)}")
    print(f"the total number of trials considered in summary is {num_trials_considered}")
    return dir_out
    #TODO: collect each diffcoeff_summary_ into one csv located in initial-conditions-2/
