from . import *
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, sys, re

def get_plotter_annihilations_a(
                            rdeath_thresh = 0.7,
                            t_col = 'tdeath',
                            id_col = 'event_id',
                            pid_col = 'pid',
                            x_col = 'r',
                            dxdt_col = 'drdt',
                            DT = 0.025,
                            DT_sec=0.000025,
                            printing = False,
                            fontsize=18,
                            **kwargs):
    '''
    Example Usage:
    plotter_annihilations_a=get_plotter_annihilations_a(**baked_kwargs)
    plotter_annihilations_a(ax,data)
    task_list.append(plotter_annihilations_a,data_unbaked_kwargs_tuple)
    '''
    def plotter_annihilations_a(ax,data):
        '''see docstring for get_plotter_annihilations_a'''
        #         navg1=41, seemed reasonable...
        #         navg2=11,
        df,valid_event_id_lst,navg1,navg2=data
        for j, event_id in enumerate(valid_event_id_lst):
            d = df[df[id_col] == event_id]
            R_values = d[x_col].values
            dRdt_values = d[dxdt_col].values
            t_values = d[t_col].values
            ax.plot(-t_values, R_values, label=j)
            if printing:
                print(f"shapes of {j}: {(t_values.shape, R_values.shape, dRdt_values.shape)}")

        #format the data
        ax.legend()
        ax.set_title(f'moving average window = {DT*navg1:.3f} ms\n(navg1={navg1})', fontsize=fontsize)
        format_plot(ax,
                    xlabel='time until death (ms)',
                    ylabel='R (cm)',
                    fontsize=fontsize,
                    use_loglog=False)
        return True
    return plotter_annihilations_a

def get_plotter_annihilations_b(
                            rdeath_thresh = 0.7,
                            t_col = 'tdeath',
                            id_col = 'event_id',
                            pid_col = 'pid',
                            x_col = 'r',
                            dxdt_col = 'drdt',
                            DT = 0.025,
                            DT_sec=0.000025,
                            printing = False,
                            fontsize=18,
                            **kwargs):
    '''
    Example Usage:
    plotter_annihilations_b=get_plotter_annihilations_a(**baked_kwargs)
    plotter_annihilations_b(ax,data)
    task_list.append(plotter_annihilations_a,data_unbaked_kwargs_tuple)
    '''
    def plotter_annihilations_b(ax,data):
        '''see docstring for get_plotter_annihilations_b'''
        #         navg1=41, seemed reasonable...
        #         navg2=11,
        df,valid_event_id_lst,navg1,navg2=data
        for j, event_id in enumerate(valid_event_id_lst):
            d = df[df[id_col] == event_id]
            R_values = d[x_col].values
            dRdt_values = d[dxdt_col].values
            t_values = d[t_col].values
            ax.plot(R_values, dRdt_values, label=j)
            if printing:
                print(f"shapes of {j}: {(t_values.shape, R_values.shape, dRdt_values.shape)}")

        #format the data
        ax.set_title(f'Savitzky-Golay window = {DT*navg2:.3f} ms\n(navg2={navg2})', fontsize=fontsize)
        format_plot(ax,
                    xlabel='R (cm)',
                    ylabel='dR/dt (cm/s)',
                    fontsize=fontsize,
                    use_loglog=False)
        return True
    return plotter_annihilations_b


#a nice example of a useful bluf generation using plotter_annihilations_a and plotter_annihilations_b
if __name__=='__main__':
    #define inputs
    input_fn='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-3-LR/param_qu_tmax_30_Ko_5.4_diffCoef_0.0005_dt_0.025/annihilations_mindur_150_maxdur_150_minrange_0.5_rangethresh_0.1.csv'
    navg1_values=np.arange(1,100,10)
    navg2_values=np.arange(5,100,10)

    #get plotter functions
    plotter_annihilations_a=get_plotter_annihilations_a(
                                rdeath_thresh = 0.7,t_col = 'tdeath',id_col = 'event_id',
                                pid_col = 'pid',x_col = 'r',dxdt_col = 'drdt',
                                DT = 0.025,DT_sec=0.000025,printing = False,fontsize=18)
    plotter_annihilations_b=get_plotter_annihilations_b(
                                rdeath_thresh = 0.7,t_col = 'tdeath',id_col = 'event_id',
                                pid_col = 'pid',x_col = 'r',dxdt_col = 'drdt',
                                DT = 0.025,DT_sec=0.000025,printing = False,fontsize=18)

    #initialize the simple finite difference dRdt as df
    df,valid_event_id_lst=get_annihilation_df_naive(input_fn)
    data=(df,valid_event_id_lst,0,0)
    task_lst=[(plotter_annihilations_a,data),
               (plotter_annihilations_b,data)]

    #generate task_lst by plotting the previous method, which used simple difference to compute dRdt
    for navg2 in navg2_values:
        for navg1 in navg1_values:
            #generate the data
            df,valid_event_id_lst=get_annihilation_df(input_fn,navg1,navg2)
            data=(df,valid_event_id_lst,navg1,navg2)
            task_lst.append((plotter_annihilations_a,data))
            task_lst.append((plotter_annihilations_b,data))

    #gener_bluf
    bluf_fn='navgs_for_dRdt.pdf'
    bluf_dir=os.path.join(trial_folder_name,'fig')
    if not os.path.exists(bluf_dir):
        os.mkdir(bluf_dir)
    os.chdir(bluf_dir)
    bluf_dir=os.path.join(bluf_dir,bluf_fn)
    working=gener_bluf(task_lst,bluf_dir)
    plt.close()
    if working:
        print(f"file saved successfully in {bluf_dir}")
