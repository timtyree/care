from .. import *
import pandas as pd, numpy as np
from ..utils.dist_func import get_distance_L2_pbc
from .compute_mean_radial_velocities import comp_neighboring_radial_velocities_between_frames
def get_routine_for_computing_dRdt_in_bulk(width,ds,DT,
                                       use_drop_shorter_than=False,
                                       drop_shorter_than=150,
                                       round_t_to_n_digits=5,
                                       frame_min=0,
                                       num_frames_between=1,
                                       use_random_frames=False,
                                       num_random_frames=500,
                                       max_dist_thresh=5,**kwargs):
    '''
    Example Usage:
    routine_for_computing_dRdt_in_bulk=routine_for_computing_dRdt_in_bulk(width)
    '''
    distance_L2_pbc=get_distance_L2_pbc(width=width, height=width)
    dist_thresh=max_dist_thresh
    def routine_for_computing_dRdt_in_bulk(input_fn):
        #load the data
        df_raw=pd.read_csv(input_fn)
        # DT = compute_DT(df_raw, round_t_to_n_digits=round_t_to_n_digits) #ms/frame
        DS=ds/width #cm/pixel
        # print(f"time between two observations is {DT} milliseconds.")
        if use_drop_shorter_than:
            #drop any trajectories don't last longer than T_min
            df_raw=get_all_longer_than(df_raw.copy(),DT,T_min=drop_shorter_than)
        #     print(f"percent of observations not filtered by minimum duration of {drop_shorter_than} ms was {100*df.size/df_raw.size:.2f}%.")

        frame_values=np.array(sorted(set(df_raw.frame.values)))#[:-num_frames_between]#not sure if needed, bc
        boo=frame_values>=frame_min
        frame_values=frame_values[boo]
        #randomly select some frames to sample
        if use_random_frames:
            random_frames=np.random.choice(frame_values[boo],size=num_random_frames,replace=False)
            frames_input=random_frames
        else:
            frames_input=frame_values

        #for each frame considered, measure dRdt and R
        R_out_lst=[];dRdt_out_lst=[]
        # for frame in random_frames:
        frame_final=frames_input[-1]
        for frame in frames_input:
            #TODO: extend to consider all pid present in frame
            R_values, dRdt_values = comp_neighboring_radial_velocities_between_frames(df_raw,frame=frame,num_frames_between=num_frames_between,
                                    distance_L2_pbc=distance_L2_pbc,dist_thresh=dist_thresh,DS=DS,DT=DT,**kwargs)
            R_out_lst.extend(R_values)
            dRdt_out_lst.extend(dRdt_values)
        #     printProgressBar(frame + 1, frame_final, prefix = 'Progress:', suffix = 'Complete', length = 50)

        R_values=np.array(R_out_lst)
        dRdt_values=np.array(dRdt_out_lst)
        #(optional) bootstrap the output
        # dict_out=bin_and_bootstrap_xy_values(x=R_values,y=dRdt_values,xlabel='r',ylabel='drdt',bins='auto',min_numobs=None,num_bootstrap_samples=1000)
        # df_drdt=pd.DataFrame(dict_out)

        #save the output as csv
        df_out=pd.DataFrame({
            'r':R_values,
            'drdt':dRdt_values,
        })
        save_folder='/'.join(os.path.dirname(input_fn).split('/')[:-1])+f'/radial_neighbor_velocities_framemin_{frame_min}_numframesbetween_{num_frames_between}_maxdistthrsh_{max_dist_thresh}'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        os.chdir(save_folder)
        save_fn=os.path.basename(input_fn).replace('.csv',f'_drdt.csv')
        df_out.to_csv(save_fn,index=False)
        return os.path.abspath(save_fn)
    return routine_for_computing_dRdt_in_bulk
