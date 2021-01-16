import numpy as np
from numba import njit

@njit
def unstack_txt(txt):
    inVc=txt[...,0:2]
    outVc=txt[...,2:4]
    inmhjdfx=txt[...,4:10]
    outmhjdfx=txt[...,10:16]
    dVcdt=txt[...,16:18]
    return inVc,outVc,inmhjdfx,outmhjdfx,dVcdt
@njit
def stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt):
    txt=np.stack((
        inVc[...,0],
        inVc[...,1],
        outVc[...,0],
        outVc[...,1],
        inmhjdfx[...,0],
        inmhjdfx[...,1],
        inmhjdfx[...,2],
        inmhjdfx[...,3],
        inmhjdfx[...,4],
        inmhjdfx[...,5],
        outmhjdfx[...,0],
        outmhjdfx[...,1],
        outmhjdfx[...,2],
        outmhjdfx[...,3],
        outmhjdfx[...,4],
        outmhjdfx[...,5],
        dVcdt[...,0],
        dVcdt[...,1]
    )).T
    return txt