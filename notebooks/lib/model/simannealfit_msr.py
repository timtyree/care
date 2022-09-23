#simannealfit_msr.py
#Programmer: Tim Tyree
#Date: 9.23.2022
import numpy as np, pandas as pd
from scipy.optimize import dual_annealing

def comp_square_error_msr_aff_osc(x,*args):
    """phase is in radians. period is in milliseconds.  a0,a1 are in cm^2/s.

Example Usage:
x0=a0,a1,period,phase
args=t_values,msr_values,D
square_error=comp_square_error_msr_aff_osc(x0,*args)
rmse=np.sqrt(square_error/t_values.shape[0])
print(f"{rmse=}")
    """
    a0,a1,period,phase=x
    t_values,msr_values,D=args
    omega=2*np.pi/period*1e3 #Hz bc period is in ms
    msr_values_affoscillatory=4*((2*D+a0)*t_values+(a1/omega)*(np.sin(omega*t_values+phase)-np.sin(phase)))
    square_error_msr=np.sum((msr_values_affoscillatory-msr_values)**2)
    return square_error_msr

def anneal_msr_fit(t_values,msr_values,D,
                   bounds = ((0, 100), (0, 100), (50, 300), (-3.15,3.15)),
                   maxiter=1000,
                   seed=42,
                   no_local_search=True,
                   **kwargs):
    """returns fit of MSR(t') to the affine oscillatory particle model.
    if no_local_search=True, simulated annealing is used.
    if no_local_search=False, dual annealing is used.
    kwargs are passed to scipy.optimize.dual_annealing
    bounds gives the upper/lower bounds to parameter values a0 (cm^2/s), a1 (cm^2/s), period (ms), and phase (radians), respectively.

    Example Usage:
res = anneal_msr_fit(t_values,msr_values,D,
                   bounds = ((0, 100), (0, 100), (50, 300), (-3.15,3.15)),
                   maxiter=1000,seed=42, no_local_search=True)#,**kwargs)
    """
    # input: t_values, msr_values, D, Gamma
    # output: a0,a1,period,phase,rmse,alinear
    args=t_values,msr_values,D
    # np.random.seed(42)   # seed to allow replication.
    res = dual_annealing(
        comp_square_error_msr_aff_osc,
        bounds=bounds,
        args=args,
        maxiter=maxiter,
        seed=seed,
        no_local_search=no_local_search,**kwargs)
    return res

def comp_alinear(a0,a1,period,Gamma,phase):
    """computes the effective attraction coefficient of the linear particle model.
    a0,a1 are in units of alinear.
    period is in units of the mean minimum lifetime, Gamma.
    phase is in units of radians.

    Example Usage:
alinear = comp_alinear(a0,a1,period,Gamma,phase)
    """
    omega = 2*np.pi/period
    z = Gamma*omega
    alinear=a0 + a1*(np.cos(phase)-z*np.sin(phase))/(1 + z**2)
    return alinear
