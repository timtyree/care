import numpy as np

def print_fit_linear(x,y):
    """returns the max likliehood estimators and 95% confidence intervals
    that result from ordinary least squares regression applied to the
    1-dimensional numpy arrays, x and y.

    Example Usage:
dict_fit = print_fit_linear(x,y)
    """
    dict_output = compute_95CI_ols(x,y)
    m=dict_output['m']
    b=dict_output['b']
    Delta_m=dict_output['Delta_m']
    Delta_b=dict_output['Delta_b']
    Rsq=dict_output['Rsquared']
    yhat=m*x+b
    rmse=np.sqrt(np.mean((yhat-y)**2))
    print(f"m={m:.6f}+-{Delta_m:.6f}")#"; B={B:.6f}+-{Delta_B:.6f}")
    print(f"b= {b:.6f}+-{Delta_b:.6f}")
    print(f"RMSE={rmse:.4f} (N={x.shape[0]})")
    print(f"R^2={Rsq:.4f}")
    return dict_output


def compute_95CI_ols(x,y):
    '''returns the max likliehood estimators and 95% confidence intervals
    that result from ordinary least squares regression applied to the
    1-dimensional numpy arrays, x and y.

    Example Usage:
dict_output = compute_95CI_ols(x,y)
    '''
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
    Delta_m = 1.96*sm
    Delta_b = 1.96*sb
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

def comp_ols_simple(x,y):
    '''returns the max likliehood estimators
    that result from ordinary least squares regression applied to the
    1-dimensional numpy arrays, x and y.

    Example Usage:
dict_output = comp_ols_simple(x,y)
    '''
    x=x.flatten();y=y.flatten()
    n=x.shape[0]
    assert(n==y.shape[0])
    assert(n>2)
    # if not n>=8:
    #     print('Warning: CI not valid for less than 8 data points!')
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
    #Delta_m = 1.96*sm
    #Delta_b = 1.96*sb
    #compute Rsquared
    Rsquared=(SSyy-SSE)/SSyy
    #format results as a human readable dict
    dict_output={
        'm':m,
        #'Delta_m':Delta_m,
        'b':b,
        #'Delta_b':Delta_b,
        'Rsquared':Rsquared
    }
    return dict_output
