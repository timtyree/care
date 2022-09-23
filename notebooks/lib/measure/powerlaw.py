# from . import *
import numpy as np
from .compute_slope import compute_95CI_ols
# from .compute_sliding_slope import *

def comp_power_scale(B,Delta_B,m,Delta_m):
    '''compute pessemistic 95% CI of annihilation rate scale, M_fk=B_fk**m_fk'''
    #min bound
    min_bound=(B-Delta_B)**(m-Delta_m)
    # print(B**m-min_bound)
    #max bound
    max_bound=(B+Delta_B)**(m+Delta_m)
    # print(max_bound-B**m)
    Delta_M=np.max((np.abs(max_bound-B**m),np.abs(B**m-min_bound)))
    M=B**m
    return M,Delta_M

def fit_power_law(x,y):
    dict_loglog=compute_95CI_ols(np.log(x),np.log(y))
    m=dict_loglog['m']
    Delta_m=dict_loglog['Delta_m']
    dict_out=compute_95CI_ols(x,y**(1/m))
    dict_linlin=dict_out
    B=dict_linlin['m'] #this method is more accurate than using y-intercept of dict_loglog
    Delta_B=dict_linlin['Delta_m']
    Rsq=dict_loglog['Rsquared']
    return B,Delta_B,m,Delta_m,Rsq


def compute_power_rmse(x_values,y_values,m,B):
    """compute rmse of power law fits"""
    yv=(B*x_values)**m
    rmse=np.sqrt(np.mean((y_values-yv)**2))
    return rmse

def print_fit_power_law(x,y,printing=True):
    """print_fit_power_law prints fit to power law by ordinary least squares (ols).
    Delta_... indicates the radial distance to the 95% CI resulting from ols.
    print_fit_power_law returns a dictionary of floats with the following schema:

    schema:
        m: exponent
        M: magnitude
        B: y-intercept
        rmse: root mean squared error
        Rsq: R squared
        num_obs: number of observations

    Example Usage:
dict_fit = print_fit_power_law(x,y)
print_dict(dict_fit)
    """
    B,Delta_B,m,Delta_m,Rsq=fit_power_law(x,y)
    rmse=compute_power_rmse(x,y,m,B)
    M, Delta_M= comp_power_scale(B,Delta_B,m,Delta_m)
    num_obs = x.shape[0]
    if printing:
        print(f"m = {m:.6f} +/- {Delta_m:.6f}")
        print(f"M = {M:.6f} +/- {Delta_M:.6f} Hz/cm^2; B={B:.6f} +/- {Delta_B:.6f}")
        print(f"RMSE={rmse:.4f} Hz/cm^2; R^2={Rsq:.4f}; N={num_obs}")
    dict_fit=dict(
        rmse=rmse,
        Rsq=Rsq,
        num_obs=num_obs,
        m=m,
        Delta_m=Delta_m,
        M=M,
        Delta_M=Delta_M,
        B=B,
        Delta_B=Delta_B,
    )
    return dict_fit
