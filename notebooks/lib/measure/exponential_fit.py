import numpy as np
from .compute_slope import *
def fit_exponential(x,y):
    dict_linlog=compute_95CI_ols(x,np.log(y))
    m=dict_linlog['m']
    Delta_m=dict_linlog['Delta_m']
    B=dict_linlog['b']
    Delta_B=dict_linlog['Delta_b']
    Rsq=dict_linlog['Rsquared']
    return B,Delta_B,m,Delta_m,Rsq

def compute_exponential_rmse(x_values,y_values,m,B):
    """compute rmse of power law fits"""
    yv=B*np.exp(x_values*m)
    rmse=np.sqrt(np.mean((y_values-yv)**2))
    return rmse

def print_fit_exponential(x,y):
    B,Delta_B,m,Delta_m,Rsq=fit_exponential(x,y)
    rmse=compute_exponential_rmse(x,y,m,B)
    print(f"m={m:.6f}+-{Delta_m:.6f}")#"; B={B:.6f}+-{Delta_B:.6f}")
    print(f"B= {B:.6f}+-{Delta_B:.6f}")
    print(f"RMSE={rmse:.4f}")
    print(f"R^2={Rsq:.4f}")
