# PlotFlowers.py
# Tim Tyree
# 12.18.2020
from ..my_initialization import *
from .. import *
from .. import *
# from .track_tips import *
from ..utils.dist_func import *
from ..utils.utils_traj import *

def PlotFlowerTrajectories(df,col="t",width=200,height=200,fontsize=24,DS = 5/200, DT=1., jump_thresh=10., alpha=0.01,saving = False, savefig_folder=None,savefig_fn=None, chop_at_first_jump = True, ax=None,cmap="Blues", **kwargs):
    '''plot the xy trajectory for longliving tips

    Example Usage:
ax=PlotFlowerTrajectories(df,col="t",width=200,height=200,fontsize=24,DS = 5/200, DT=1., jump_thresh=10., alpha=0.01,saving = False, savefig_folder=None,savefig_fn=None, chop_at_first_jump = True, ax=None,cmap="Blues")#, **kwargs)

    '''
    pid_lst = sorted(set(df.particle.values))
    df_traj = df
    xmin=0; ymin=0; xmax=width; ymax=height;
    axis = [xmin,xmax,ymin,ymax]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    for pid in  pid_lst:#[2:]:
        x_values ,y_values, c_values = df_traj[(df_traj.particle==pid)][['x','y', col]].values.T
        if chop_at_first_jump:
            jump_index_array, spd_lst = find_jumps(x_values,y_values,width=width,height=height, DS=DS,DT=DT, jump_thresh=jump_thresh)#.25)
            if len(jump_index_array)>0:
                ji = jump_index_array[0]
                x_values = x_values[:ji]
                y_values = y_values[:ji]
                c_values = c_values[:ji]
        #scale to real coords
        x_values *= DS
        y_values *= DS
        plt.scatter(x_values,y_values, s=20,#s=0.1
                    c=c_values, vmin = np.min(c_values), vmax = np.max(c_values), cmap=cmap,alpha=alpha)#, **kwargs)
        plt.scatter([x_values[0]],[y_values[0]], s=600,color='green',marker='*')
        plt.scatter([x_values[-1]],[y_values[-1]], s=600,color='red',marker='*')
    # plt.axis(axis)
    # plt.title(f'''more time = more blue''', fontsize=fontsize)
    #plt.title(f'''more {col} = more blue''', fontsize=fontsize)
    ax.set_xlabel('x (cm)', fontsize=fontsize)
    ax.set_ylabel('y (cm)', fontsize=fontsize)
    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=0)
    ax.grid('on')
    ax.set_aspect('equal')
    if not saving:
#         plt.show()
        return ax
    else:
        plt.tight_layout()
        os.chdir(savefig_dir)
        plt.savefig(savefig_fn, dpi=300)
        plt.close()
        print(f"saved figure in \n\t{savefig_fn}.")
        return savefig_fn


# def PlotFlowerTrajectories(input_file_name, n_tips=1,col="t",width=200,height=200,saving = True,fontsize=24,DS = 5/200, DT=1., jump_thresh=10., alpha=0.01):
#     '''plot the xy trajectory for longliving tips'''
#     V_thresh = eval(input_file_name[input_file_name.find('_V_')+len('_V_'):].split('_')[0])
#     savefig_dir = os.path.join(nb_dir,f'Figures/flower-plots/V_{V_thresh}')
#     savefig_fn = os.path.basename(input_file_name).replace('.csv',f'_ntips_{n_tips}_longest_tips.png')
#     xmin=0; ymin=0; xmax=width; ymax=height;
#     axis = [xmin,xmax,ymin,ymax]
#     fig, ax = plt.subplots(figsize=(8,8))
#     for pid in  pid_longest_lst:#[2:]:
#         x_values ,y_values, c_values = df_traj[(df_traj.particle==pid)][['x','y', col]].values.T
#         jump_index_array, spd_lst = find_jumps(x_values,y_values,width=width,height=height, DS=DS,DT=DT, jump_thresh=jump_thresh)#.25)
#         if len(jump_index_array)>0:
#             ji = jump_index_array[0]
#             x_values = x_values[:ji]
#             y_values = y_values[:ji]
#             c_values = c_values[:ji]
#
#         #scale to real coords
#         x_values *= DS
#         y_values *= DS
#         plt.scatter(x_values,y_values, s=20,#s=0.1
#                     c=c_values, vmin = np.min(c_values), vmax = np.max(c_values), cmap="Blues",alpha=alpha)
#         plt.scatter([x_values[0]],[y_values[0]], s=40,color='green')
#         plt.scatter([x_values[-1]],[y_values[-1]], s=40,color='red')
#     # plt.axis(axis)
#     # plt.title(f'''more time = more blue''', fontsize=fontsize)
#     #plt.title(f'''more {col} = more blue''', fontsize=fontsize)
#     ax.set_xlabel('x (cm)', fontsize=fontsize)
#     ax.set_ylabel('y (cm)', fontsize=fontsize)
#     # We change the fontsize of minor ticks label
#     ax.tick_params(axis='both', which='major', labelsize=fontsize)
#     ax.tick_params(axis='both', which='minor', labelsize=0)
#     ax.grid('on')
#     ax.set_aspect('equal')
#     if not saving:
#         plt.show()
#     else:
#         plt.tight_layout()
#         os.chdir(savefig_dir)
#         plt.savefig(savefig_fn, dpi=300)
#         plt.close()
#         print(f"saved figure in \n\t{savefig_fn}.")
#     return savefig_fn
