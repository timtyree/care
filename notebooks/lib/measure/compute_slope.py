import numpy as np

def compute_95CI_ols(x,y):
    '''returns the max likliehood estimators and 95% confidence intervals
    that result from ordinary least squares regression applied to the
    1-dimensional numpy arrays, x and y.'''
    x=x.flatten();y=y.flatten()
    n=x.shape[0]
    assert(n==y.shape[0])
    assert(n>2)
    if not n>=8:
        print('Warning: CI not valid for less than 8 data points!')
    xbar=np.mean(x);ybar=np.mean(y)
    #compute sums of squares
    SSxx=np.sum((x-xbar)**2)
    SSxy=np.dot((x-xbar),(y-ybar))
    SSyy=np.sum((y-ybar)**2)
    #best linear unbiased estimator of slope
    m=SSxy/SSxx
    #best linear unbiased estimator of intercept
    b=ybar-m*xbar
    #values of fit
    yhat=b+m*x
    #standard error of fit, s_{y,x}^2=ssE
    SSE=np.sum((y-yhat)**2)
    ssE=SSE/(n-2)
    #standard deviation of slope
    sm = np.sqrt(ssE/SSxx)
    #standard deviation of intercept
    sb = np.sqrt(ssE*(1/n+xbar**2/SSxx))
    #compute 95% CI for parameters
    Delta_m = 2*sm
    Delta_b = 2*sb
    #compute Rsquared
    Rsquared=(SSyy-SSE)/SSyy
    #format results as a human readable dict
    dict_output={
        'm':m,
        'Delta_m':Delta_m,
        'b':b,
        'Delta_b':Delta_b,
        'Rsquared':Rsquared
    }
    return dict_output
