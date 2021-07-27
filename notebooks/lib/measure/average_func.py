import numpy as np

def geo_mean_overflow(iterable):
    '''This method is both fast and avoids numerical error by averaging the log of the arguments
    and then performing the exponential map.'''
    a = np.log(iterable)
    return np.exp(a.mean())
