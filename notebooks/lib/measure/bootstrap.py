import numpy as np, pandas as pd
from scipy.stats import normaltest

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
    mean_values=bootstrap_mean(x,num_samples=num_samples)
    sig=np.std(mean_values)
    _, p = normaltest(mean_values)
    Delta_mean=1.96*sig
    return Delta_mean,p
