import numpy as np, pandas as pd
from ..measure.bootstrap import *

def comp_mean_radial_velocities(df,t_col='tdeath',bins='auto',min_numobs=None,num_samples=1000):
    '''computes the mean radial velocities, binning by radius.
    supposes df is from a .csv file containing annihilation or creation results,
    where rows are presorted according to event and then by t_col.
    returns a dict containing results for mean radial velocities.
    if min_numobs is None, then min_numobs is determined from the mean counts in each bin.
    Example Usage:
    dict_out=compute_mean_radial_velocities(df,t_col='tdeath')
    '''
    DT=sorted(set(df[t_col].values))[0]
    assert(DT>0)#if DT<0, then a factor of -1 is needed in a few places...
    df['drdt']=df['r'].diff()/DT
    #set drdt to zero where pid changes or where tdeath jumps by more than dt
    boo=df[t_col].diff()!=-DT
    df.loc[boo,'drdt']=np.nan
    df.dropna(inplace=True)

    #implement measure of dRdt that explicitely bins by radius
    counts,r_edges=np.histogram(df.r.values,bins=bins)
    range_values=r_edges
    if min_numobs is None:
        min_numobs=np.mean(counts)/8
    r_lst=[];drdt_lst=[];Delta_r_lst=[];Delta_drdt_lst=[];
    count_lst=[];p_r_lst=[];p_drdt_lst=[]
    for j in range(r_edges.shape[0]-1):
        numobs=counts[j]
        if numobs>min_numobs:
            boo=(df.r>=r_edges[j])&(df.r<r_edges[j+1])
            dfb=df[boo]
            r_values=dfb.r.values
            drdt_values=dfb.drdt.values
            #compute mean values in bin
            r=np.mean(r_values)
            drdt=np.mean(drdt_values)
            # compute 95% CI for mean
            Delta_r,p_r=bootstrap_95CI_Delta_mean(r_values,
                                                 num_samples=num_samples)
            Delta_drdt,p_drdt=bootstrap_95CI_Delta_mean(drdt_values,
                                                 num_samples=num_samples)
            #append results to list
            r_lst.append(r)
            drdt_lst.append(drdt)
            Delta_r_lst.append(Delta_r)
            Delta_drdt_lst.append(Delta_drdt)
            p_r_lst.append(p_r)
            p_drdt_lst.append(p_drdt)
            count_lst.append(numobs)
    r_values=np.array(r_lst)
    drdt_values=np.array(drdt_lst)
    Delta_r_values=np.array(Delta_r_lst)
    Delta_drdt_values=np.array(Delta_drdt_lst)
    p_r_values=np.array(p_r_lst)
    p_drdt_values=np.array(p_drdt_lst)
    count_values=np.array(count_lst)
    dict_out={
        'r':r_values,
        'drdt':drdt_values,
        'Delta_r':Delta_r_values,
        'Delta_drdt':Delta_drdt_values,
        'p_r':p_r_values,
        'p_drdt':p_drdt_values,
        'counts':count_values
    }
    return dict_out

def save_mean_radial_velocities(input_fn,t_col='tdeath',output_fn=None,bins='auto'):
    if output_fn is None:
        output_fn=input_fn.replace('.csv',f'_mean_radial_velocities_bins_{bins}.csv')

    df=pd.read_csv(input_fn)
    dict_out=comp_mean_radial_velocities(df,t_col=t_col,bins=bins)
    df_drdt=pd.DataFrame(dict_out)
    df_drdt.to_csv(output_fn,index=False)
    return output_fn
