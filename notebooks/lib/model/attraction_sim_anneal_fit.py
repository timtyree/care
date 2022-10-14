# attraction_sim_anneal_fit.py
#Programmer: Tim Tyree
#Date: 10.7.2022
import numpy as np, pandas as pd, sys, os
from scipy.optimize import dual_annealing
from ..measure.bootstrap import bootstrap_95CI_Delta_mean

def comp_mean_bootstrap_uncertainty(x,num_samples=1000):
    """
    Example Usage:
meanx,Delta_meanx,num_obs,p_normal=comp_mean_bootstrap_uncertainty(minlifetime_values)
printing=True
if printing:
    print(f"mean: {meanx:.4f} +/- {Delta_meanx:.4f} (N={num_obs}, {p_normal=:.4f})")
    """
    meanx = np.mean(x)
    Delta_meanx,p_normal=bootstrap_95CI_Delta_mean(x,num_samples=num_samples)
    num_obs=x.shape[0]
    return meanx,Delta_meanx,num_obs,p_normal

def comp_square_error_msr_aff_osc_period_fixed_with_msr_offset(x,*args):
    """phase is in radians. period is in milliseconds.  a0,a1 are in cm^2/s.
    MSR_offset is in cm^2

Example Usage:
x0=a0,a1,phase
period=T*1e3 #bc period is in milliseconds
args=t_values,msr_values,D,period
square_error=comp_square_error_msr_aff_osc_period_fixed(x0,*args)
rmse=np.sqrt(square_error/t_values.shape[0])
print(f"{rmse=}")
    """
    a0,a1,phase,MSR_offset=x
    t_values,msr_values,D,period=args
    omega=2*np.pi/period*1e3 #Hz bc period is in ms
    msr_values_affoscillatory=4*((2*D+a0)*t_values+(a1/omega)*(np.sin(omega*t_values+phase)-np.sin(phase)))+MSR_offset
    square_error_msr=np.sum((msr_values_affoscillatory-msr_values)**2)
    return square_error_msr


def comp_square_error_msr_aff_osc_period_fixed(x,*args):
    """phase is in radians. period is in milliseconds.  a0,a1 are in cm^2/s.

Example Usage:
x0=a0,a1,phase
period=T*1e3 #bc period is in milliseconds
args=t_values,msr_values,D,period
square_error=comp_square_error_msr_aff_osc_period_fixed(x0,*args)
rmse=np.sqrt(square_error/t_values.shape[0])
print(f"{rmse=}")
    """
    a0,a1,phase=x
    t_values,msr_values,D,period=args
    omega=2*np.pi/period*1e3 #Hz bc period is in ms
    msr_values_affoscillatory=4*((2*D+a0)*t_values+(a1/omega)*(np.sin(omega*t_values+phase)-np.sin(phase)))
    square_error_msr=np.sum((msr_values_affoscillatory-msr_values)**2)
    return square_error_msr

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

#################################
def fit_msr_oscillatory(x_values,y_values,max_tdeath,
                        D=0., #cm^2/s
                        tscale=1e-3, #s per ms
                        maxiter=10000,
                        bounds = ((0, 100), (0, 100), (50, 300), (-3.15,3.15)),
                        seed=42,
                        no_local_search=True,
                        printing=True,
                        **kwargs):
    """inputed xy values correspond to tdeath,msd, respectively.
    bounds is the bounding box in the linear basis of (a0,a1,period,phase), respectively.
    kwargs are passed to anneal_msr_fit directly.

    Example Usage: print fit of oscillatory particle model to full
a0,a1,period,phase,rmse = fit_msr_oscillatory(x_values,y_values,max_tdeath,
                                    D=0.,maxiter=10000,printing=True)#,**kwargs)
dict_fit=dict(a0=a0,a1=a1,period=period,phase=phase,rmse=rmse)
print_dict(dict_fit)
    """
    boo=x_values<max_tdeath
    t_values=x_values[boo].copy()*tscale
    msr_values=y_values[boo].copy()
    msr_values-= np.min(msr_values) #corrects for aliasing
    if printing:
        print(f"performing simulated annealing with {D=:.0f} fixed (num. epochs: {maxiter})...")
    res = anneal_msr_fit(t_values,msr_values,D,
                       bounds = bounds,
                       maxiter=maxiter,seed=seed, no_local_search=no_local_search,**kwargs)
    rmse= np.sqrt(res.fun/t_values.shape[0])
    a0,a1,period,phase=res.x
    if printing:
        print(f"simulated annealing fit: {a0=:.4f}, {a1=:.4f}, {period=:.4f}, {phase=:.4f} --> {rmse=:.4f} cm^2")
    rmse= np.sqrt(res.fun/t_values.shape[0])
    return a0,a1,period,phase,rmse
