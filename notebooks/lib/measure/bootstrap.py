import numpy as np, pandas as pd, dask.bag as db
from scipy.stats import normaltest
import time

def test_normality(x,alpha = 0.05):
    '''performs D'Agostino and Pearson's omnibus test for normality.
    Returns p, True if significantly different from normal distribution'''
    _, p = normaltest(x)
    is_significant = p < alpha
    return p, is_significant

def print_test_normality(x,alpha = 0.05):
    _, p = normaltest(x)
    is_significant = p < alpha
    print(f"p = {p:.10g}")
    if is_significant:  # null hypothesis: x comes from a normal distribution
        print("\tThe null hypothesis can be rejected.  The data is significantly different from the normal distribution.")
    else:
        print("\tThe null hypothesis cannot be rejected.  The data is not significantly different from the normal distribution.")

def bootstrap_mean(x,num_samples=1000):
    mean_values=np.zeros(num_samples)
    sizex=x.shape[0]
    for i in range(num_samples):
        randint_values=np.random.randint(low=0, high=sizex, size=sizex, dtype=int)
        x_bootstrap=x[randint_values]
        mean_values[i]=np.mean(x_bootstrap)
    return mean_values

def bootstrap_stdev_of_mean(x,num_samples=1000):
    mean_values=bootstrap_mean(x,num_samples=num_samples)
    sig=np.std(mean_values)
    return sig

def bootstrap_95CI_Delta_mean(x,num_samples=1000):
    """
    Example Usage:
    Delta_mean,p=bootstrap_95CI_Delta_mean(x,num_samples=1000)
    """
    mean_values=bootstrap_mean(x,num_samples=num_samples)
    sig=np.std(mean_values)
    _, p = normaltest(mean_values)
    Delta_mean=1.96*sig
    return Delta_mean,p

def bin_and_bootstrap_xy_values(x,y,xlabel,ylabel,bins='auto',min_numobs=None,num_bootstrap_samples=1000,npartitions=1,**kwargs):
    '''see docstring for bin_and_bootstrap_xy_values_parallel'''
    type_in=type(np.ndarray([]))
    if type(x)!=type_in:
        raise "InputError: x and y must have type np.ndarray!"
    if type(y)!=type_in:
        raise "InputError: x and y must have type np.ndarray!"
    R_values=x
    dRdt_values=y
    num_samples=num_bootstrap_samples
    #implement measure of dRdt that explicitely bins by radius
    counts,r_edges=np.histogram(R_values,bins=bins)
    range_values=r_edges
    if min_numobs is None:
        min_numobs=np.mean(counts)/8

    r_lst=[];drdt_lst=[];Delta_r_lst=[];Delta_drdt_lst=[];
    count_lst=[];p_r_lst=[];p_drdt_lst=[]
    if npartitions==1:
        #for a single core in base python
        for j in range(r_edges.shape[0]-1):
            numobs=counts[j]
            if numobs>min_numobs:
                boo=(R_values>=r_edges[j])&(R_values<r_edges[j+1])
                r_values=R_values[boo]
                drdt_values=dRdt_values[boo]
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
            xlabel:r_values,
            ylabel:drdt_values,
            f'Delta_{xlabel}':Delta_r_values,
            f'Delta_{ylabel}':Delta_drdt_values,
            f'p_{xlabel}':p_r_values,
            f'p_{ylabel}':p_drdt_values,
            'counts':count_values
        }
        return pd.DataFrame(dict_out)
    else:
        #perform in parallel on multiple cores
        return bin_and_bootstrap_xy_values_parallel(x,
                                       y,
                                       xlabel,
                                       ylabel,
                                       bins=bins,
                                       min_numobs=min_numobs,
                                       num_bootstrap_samples=num_bootstrap_samples,
                                       npartitions=npartitions,**kwargs)

###########################################
# Parallel implementation on multiple cores
###########################################
def get_routine_bootstrap_bin(x_values,y_values,x_bin_edges,counts,num_samples=100,min_numobs=100,**kwargs):
    '''x_values,y_values,x_bin_edges are 1 dimensional numpy arrays.
    returns the function, routine_bootstrap_bin.'''
    type_in=type(np.ndarray([]))
    if type(x_values)!=type_in:
        raise "InputError: x and y must have type np.ndarray!"
    if type(y_values)!=type_in:
        raise "InputError: x and y must have type np.ndarray!"
    R_values=x_values
    dRdt_values=y_values
    r_edges=x_bin_edges
    def routine_bootstrap_bin(bin_id):
        j=bin_id
        numobs=counts[j]
        if numobs>min_numobs:
            boo=(R_values>=r_edges[j])&(R_values<r_edges[j+1])
            r_values=R_values[boo]
            drdt_values=dRdt_values[boo]
            #compute mean values in bin
            r=np.mean(r_values)
            drdt=np.mean(drdt_values)
            # compute 95% CI for mean
            Delta_r,p_r=bootstrap_95CI_Delta_mean(r_values,
                                                 num_samples=num_samples)
            Delta_drdt,p_drdt=bootstrap_95CI_Delta_mean(drdt_values,
                                                 num_samples=num_samples)
            return np.array((r,drdt,Delta_r,Delta_drdt,p_r,p_drdt,numobs))
        else:
            return None
    return routine_bootstrap_bin

def bin_and_bootstrap_xy_values_parallel(x,
                               y,
                               xlabel='x',
                               ylabel='y',
                               bins='auto',
                               min_numobs=None,
                               num_bootstrap_samples=1000,
                               npartitions=1,
                               use_test=True,
                               test_val=0,printing=False,**kwargs):
    '''
    bin_and_bootstrap_xy_values_parallel returns a pandas.DataFrame instance with the following columns
    columns=[xlabel,ylabel,f'Delta_{xlabel}',f'Delta_{ylabel}',f'p_{xlabel}',f'p_{ylabel}','counts']
    Output Field for a given x bin include:
    - y       : the simple mean value of y
    - Delta_y : difference between the mean value and the edge of the 95% confidence interval of the mean value
    - p_y     : p value estimating the liklihood to reject the null hypothesis which claims the boostrapped distribution is normally distributed.
    - counts  : the number of observations in this bin

    Arguments x and y are assumed to be 1 dimensional numpy.array instances.
    If p_y reliably has values less than 0.05, then consider increasing num_bootstrap_samples

    bin_and_bootstrap_xy_values_parallel passes the kwarg, bins, to numpy.histogram
    to generate x bins, it then passes each x bin to npartitions dask workers, each of which
    selects the y values that correspond to the x values within its respective bin.  With these
    x and y values in hand, that worker bootstraps num_bootstrap_samples samples of approximations of
    the mean value for those x and y values.

    A bin is ignored if it contains no more than min_numobs
    observations.  If min_numobs=None, then min_numobs=np.mean(counts)/8, where counts is the array of
    counts in each bin.

    Example Usage: compute the bootstrapped mean dRdt (y_values) binning by one_over_R (x_values)
boo=x_values>0
df_out=bin_and_bootstrap_xy_values_parallel(x=x_values[boo],
                               y=y_values[boo],
                               xlabel='one_over_R',
                               ylabel='dRdt',
                               bins='auto',
                               min_numobs=None,
                               num_bootstrap_samples=1000,
                               npartitions=os.cpu_count(),#full cylinders
                               use_test=False,
                               test_val=0,printing=False)#,**kwargs)

    '''
    x_values=x
    y_values=y
    num_samples=num_bootstrap_samples
    counts,x_bin_edges=np.histogram(x_values,bins=bins)
    bin_id_lst=list(range(x_bin_edges.shape[0]-1))
    if min_numobs is None:
        min_numobs=np.mean(counts)/8

    #bake method to bootstrap 95%CI of mean of y conditioned on x being within a given bin
    routine_bootstrap_bin=get_routine_bootstrap_bin(x_values,y_values,x_bin_edges,counts,num_samples=num_samples,min_numobs=min_numobs)
    def routine(input_val):
        try:
            return routine_bootstrap_bin(input_val)
        except Exception as e:
            return f"Warning: something went wrong, {e}"

    #optionally test the routine
    if use_test:
        retval=routine(test_val)

    #all CPU version
    b = db.from_sequence(bin_id_lst, npartitions=npartitions).map(routine)
    start = time.time()
    retval = list(b)
    if printing:
        print(f"run time for bootstrapping was {time.time()-start:.2f} seconds.")

    array_out=np.stack([x for x in retval if x is not None])
    columns=[xlabel,ylabel,f'Delta_{xlabel}',f'Delta_{ylabel}',f'p_{xlabel}',f'p_{ylabel}','counts']
    # columns=['r','drdt','Delta_r','Delta_drdt','p_r','p_drdt','counts']
    df=pd.DataFrame(data=array_out,columns=columns)
    df=df.astype({'counts': 'int32'})
    return df
