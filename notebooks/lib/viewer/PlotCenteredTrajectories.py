# Tim Tyree
# 12.18.2020

def plot_centered_trajectories(df_traj,pid_longest_lst,width,height,saving = True,DS = 5/200,fontsize=24,col = "t",dpi=300,
    savefig_folder):
    #plot the xy trajectory for longliving tips
    # savefig_folder = os.path.join(nb_dir,'Figures/msd')
    n_tips = len(pid_longest_lst)
    savefig_fn = os.path.basename(input_file_name).replace('log.csv',f'{n_tips}_longest_tips.png')
    xmin=0; ymin=0; xmax=width; ymax=height;
    axis = [xmin,xmax,ymin,ymax]
    fig, ax = plt.subplots(figsize=(8,8))
    for pid in  pid_longest_lst:#[2:]:
        x_values ,y_values, c_values = df_traj[(df_traj.particle==pid)][['x','y', col]].values.T
        #scale to real coords
        x_values *= DS
        y_values *= DS
        plt.scatter(x_values,y_values, s=20,#s=0.1
                    c=c_values, vmin = np.min(c_values), vmax = np.max(c_values), cmap="Blues",alpha=0.01)
        plt.scatter([x_values[0]],[y_values[0]], s=40,color='green')
        plt.scatter([x_values[-1]],[y_values[-1]], s=40,color='red')
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
        plt.show()
    else:
        plt.tight_layout()
        os.chdir(savefig_folder)
        plt.savefig(savefig_fn, dpi=300)
        # print(f"saved figure in \n\t{savefig_fn}.")
        del ax
        del fig
