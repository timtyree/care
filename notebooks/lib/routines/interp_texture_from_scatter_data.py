import numpy as np
import os
from sklearn import linear_model
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator


def comp_meshgrid_from_X(X,nsamples=1000):
    '''define the local grid for visualization of N-dim interpolation'''
    num_cols=X.shape[1]
    xi=[]
    for i in range(num_cols):
        x=X[:,i]
        xi.append( np.linspace(np.min(x), np.max(x),nsamples) )
    XI_lst=np.meshgrid(*xi)
    return XI_lst

def comp_coordinate_values_from_X(X,nsamples=1000):
    XI_lst=comp_meshgrid_from_X(X,nsamples=nsamples)
    x1_values=XI_lst[0]
    x2_values=XI_lst[1]
    return x1_values, x2_values

def interp_txt_from_scatter(X,y,nsamples=1000,mode ='spline',**kwargs):
    '''input: X,y,nsamples=1000:
        - (N,2) numpy array as X, (N,1) numpy array as y
        - mode='spline' uses CloughTocher2DInterpolator, which recieves kwargs
        - mode='linear' uses LinearNDInterpolator, which recieves kwargs
    output: x1_values,x2_values,y_values

    Example Usage:
    x1_values,x2_values,y_values=interp_txt_from_scatter(X,y)
    '''
    if mode == 'linear':
        interp = LinearNDInterpolator(X, y,**kwargs)
    elif mode == 'spline':
        interp = CloughTocher2DInterpolator(X, y,**kwargs)
    else:
        pass
        raise(f"Warning: mode={mode} is not yet implemented!")

    #compute the xy coordinate images
    x1_values, x2_values = comp_coordinate_values_from_X(X,nsamples=nsamples)

    #compute the target value images
    gridshape=x1_values.shape
    X_values =np.array(list(zip((x1_values.flatten(),x2_values.flatten()))))[:,0,:].T
    y_values = interp(X_values).reshape(gridshape[0],gridshape[1])
    return x1_values,x2_values,y_values
