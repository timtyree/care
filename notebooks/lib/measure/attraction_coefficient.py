import numpy as np, pandas as pd
from .compute_slope import compute_95CI_ols
from .bootstrap import bootstrap_95CI_Delta_mean

def comp_a_SR(tdeath_values,SR_values,tmax,tscale=0.001):
    '''
    default tscale scales cm^2/milliseconds to cm^2/seconds in the output attraction coefficient value.  no scaling is performed on the length parameter in SR_values.
    Example Usage:
    a_SR, Delta_a_SR, Rsq_a_SR = comp_a_SR(tdeath_values,SR_values,tmax)
    '''
    boo=tdeath_values<tmax
    x=tdeath_values[boo]*tscale #s
    y=SR_values[boo]# cm^2
    dict_force_fit=compute_95CI_ols(x,y)
    a_SR=dict_force_fit['m']/4
    Delta_a_SR=dict_force_fit['Delta_m']/4
    Rsq_a_SR=dict_force_fit['Rsquared']
    return a_SR, Delta_a_SR, Rsq_a_SR

def routine_boostrap_tdeath_group(tdeath):
    x=df_r.loc[df_r['tdeath']==tdeath,'SR'].values
    #Delta_x, p_x = bootstrap_95CI_Delta_mean(x,num_samples=100)
    Delta_x, p_x = bootstrap_95CI_Delta_mean(x,num_samples=1000)
    return (tdeath,Delta_x, p_x)
