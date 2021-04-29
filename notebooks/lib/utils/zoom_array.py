import numpy as np
# zooming in for a square computational domain
# Programmer: Tim Tyree
# Date: 4.29.2021

def zoomin_txt(txt_in):
    '''returns txt with spatial resolution
     doubled, enforcing periodic boundary
     conditions
     '''
    width,height,chnlno=txt_in.shape[:3]
    out=np.zeros((2*width,2*height,chnlno))
    #linearly interpolate
    out[::2,::2,:]=txt_in.copy()
    out[1:-2:2]=(out[0:-2:2]+out[2:-1:2])/2
    out[:,1:-2:2]=(out[:,0:-2:2]+out[:,2:-1:2])/2
    #final row/column for pbc
    out[-1,:]=(out[-2,:]+out[0,:])/2
    out[:,-1]=(out[:,-2]+out[:,0])/2
    return out
