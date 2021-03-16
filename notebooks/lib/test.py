df['src']=df['particle']

t_values, msd_values, std_values = compute_average_std_msd(df,DT)

    ax.fill_between(t_values,msd_values-std_values,msd_values+std_values,color=colorB, alpha=0.3,step='post')
    ax.plot(t_values,msd_values,c=colorB,lw=2)

    dict_out_lst=[compute_emsd_for_longest_trajectories(input_file_name, n_tips=n_tips,DS=DS,DT=DT,L=L) for input_file_name in file_name_list]
    if len(dict_out_lst)==0:
        print(f"""no sufficiently long lasting trajectory was found.  returning None, None for
        input_file_name, {input_file_name}.""")
        return None, None
    df = pd.concat(dict_out_lst)
    df.reset_index(inplace=True,drop=True)
    #save results
    dirname = os.path.dirname(input_file_name).split('/')[-1]
    folder_name=os.path.dirname(input_file_name)
    if save_folder is None:
        save_folder = folder_name.replace(dirname,'msd')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if output_file_name is None:
        output_file_name = f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv"
    os.chdir(save_folder)
    df.to_csv(output_file_name, index=False)

    #compute average msd by trial for a subset of trials
    src_lst = sorted(set(df.src.values))
    # src_lst = src_lst#[:10]
    ff = df.copy()#pd.concat([df[df.src==src] for src in src_lst])
    # dt = DT/10**3 #seconds per frame
    # t_values = np.array(sorted(set(ff.lagt.values)))
    # t_values = np.arange(np.min(t_values),np.max(t_values),dt)#no floating point error

    t_values, msd_values, std_values = compute_average_std_msd(df,DT)
