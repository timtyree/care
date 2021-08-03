# compute_interactions.py
from ..my_initialization import *
from ..utils.utils_traj import *



def compute_df_interactions(input_fn,DS=5./200.,width=200,height=200,tmin=100,pid_col='particle',t_col='t',min_duration=150,**kwargs):
    '''input_file_name is a .csv of spiral tip trajecotries.
    filters time before tmin'''
    distance_L2_pbc = get_distance_L2_pbc(width=width,height=height)
    #list of length sorted trajectories
    df = pd.read_csv(input_fn)
    using_particle=True
    if using_particle:
        df['cid']=df[pid_col]
    df = df[df.t>tmin].copy()
    df.reset_index(inplace=True)
    s = df.groupby(pid_col).t.count()
    s = s.sort_values(ascending=False)
    pid_longest_lst = list(s.index.values)#[:n_tips])

    #TODO: add minimum duration filtering here
    #TODO: add minimum duration filtering here
    #print summary stats on particle lifetimes for one input folder
    dft=df.groupby(pid_col)[t_col].describe()
    df_lifetimes=-dft[['max','min']].T.diff().loc['min']
    # if printing:
    #     print(f"termination time was {df[t_col].max():.2f} ms")
    boo=df_lifetimes>=min_duration
    pid_longest_lst=sorted(boo[boo].index.values)

    #compute lifetime_of_sibling
    r0_lst = []; rT_lst=[]; Tdiff_lst = []; Tavg_lst = []; pid_lst = []; pid_other_lst = []; pid_death_lst=[]
    for pid in pid_longest_lst:
        # pid = pid_longest_lst[0]
        # - DONE: identify the birth mate of a given spiral tip
        d = df[df[pid_col] == pid]
        #identify the death partner
        # nearest_pid, reaction_distance_death, t_of_death = identify_death_partner(df=f,pid=pid)
        #identify the birth partner of that given tip
        pid_partner, reaction_distance_birth, t_of_life = identify_birth_partner(df=df,cid=pid,distance_L2_pbc=distance_L2_pbc,pid_col=pid_col)
        pid_partner_death, reaction_distance_death, t_of_death = identify_death_partner(df=df,cid=pid,distance_L2_pbc=distance_L2_pbc,pid_col=pid_col)
        d_other = df[df[pid_col]==pid_partner]

        # compute lifetimes of ^those spiral tips. compute average_lifetime.
        absdiff,avgval=comp_lifetime_diff_and_avg(d,d_other)

        r0_lst.append (  float(reaction_distance_birth)  )
        rT_lst.append (  float(reaction_distance_death)  )
        Tdiff_lst.append  (  float(absdiff)  )
        Tavg_lst.append  (  float(avgval)   )
        pid_lst.append  ( int(pid) )
        pid_other_lst.append  (  int(pid_partner)  )
        pid_death_lst.append  (  int(pid_partner_death))

    df_out = pd.DataFrame({
        'pid':pid_lst,
        'pid_birthmate':pid_other_lst,
        'pid_deathmate':pid_death_lst,
        'r0':r0_lst,
        'rT':rT_lst,
        'Tavg':Tavg_lst,
        'Tdiff':Tdiff_lst
    })
    df_interactions = df_out.copy()
    return df_interactions
