# compute_interactions.py
from ..my_initialization import *
def compute_df_interactions(input_file_name,DS=5./200.):
    '''input_file_name is a .csv of spiral tip trajecotries'''
    #list of length sorted trajectories
    df = pd.read_csv(input_file_name)
    df = df[df.t>100].copy()
    df.reset_index(inplace=True)
    s = df.groupby('particle').t.count()
    s = s.sort_values(ascending=False)
    pid_longest_lst = list(s.index.values)#[:n_tips])
    #compute lifetime_of_sibling
    r0_lst = []; rT_lst=[]; Tdiff_lst = []; Tavg_lst = []; pid_lst = []; pid_other_lst = []; pid_death_lst=[]
    for pid in pid_longest_lst:
        # pid = pid_longest_lst[0]
        # - DONE: identify the birth mate of a given spiral tip
        d = df[df.particle == pid]
        #identify the death partner
        # nearest_pid, reaction_distance_death, t_of_death = identify_death_partner(df=f,pid=pid)
        #identify the birth partner of that given tip
        pid_partner, reaction_distance_birth, t_of_life = identify_birth_partner(df=df,pid=pid)
        pid_partner_death, reaction_distance_death, t_of_death = identify_death_partner(df=df,pid=pid)
        d_other = df[df.particle==pid_partner]

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
