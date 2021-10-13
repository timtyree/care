#!/usr/bin/env python3
#Programmer: Tim Tyree
#Date: 10.13.2021
import numpy as np

def generate_lookup_table_LR_I(
        ndim=2500,
        dt=0.01,
        dv=0.1,
        K_o=5.4,
        dtype=np.float64,
        order='F',
        **kwargs):
    r"""generate_lookup_table_LR_I determines the look-up table for the LR-I model
    to generate lookup tables for a time step of dt=0.01 ms from terminal, run
    python3 gener_table_LR_I.py 0.01

    generate_lookup_table_LR_I returns a list of a lookup-tables that map a voltage in the first column to precomputed values in the following columns.
    these precomputed valures are needed to compute the transmembrane ion current.

    Parameters
    ----------
    ndim : int
        number of rows in the lookup table
    dt   : float
        the size of the timestep in milliseconds
    dv   : float
        the voltage between two rows in the lookup table in millivolts
    K_o  : float
        the potassium concentration outside of the myocardial tissue in millimolar
    dtype: numpy type, optional
        the data type of the output lookup table.  default is double floating point precision.
    order: str, optional
        the ordering of the output array. F is for Fortran.  C is for C programming.
    **kwargs
        arbitrary optional keyword arguments

    Returns
    -------
    tuple of numpy arrays
        each numpy array is a lookup-table, which are defined as follows

        arr10=np.stack((v_values,xinfh,xttab))
        arr11=np.stack((v_values,e1,em,ef))
        arr12=np.stack((v_values,ed,ej,eh))
        arr13=np.stack((v_values,xtaud,xtauf))
        arr39=np.stack((v_values,xinf1,xtau1,xinfm,xtaum,xinfh,xtauh,xinfj,xtauj,\
             xinfd,xtaud,xinff,xtauf,xttab,x1,e1,em,eh,ej,ed,ef))
        return (arr10,arr11,arr12,arr13,arr39)

    """
    #preallocate memory, enforce ordering, and enforce data type
    xinf1=np.zeros(ndim, dtype=dtype, order=order)
    xtau1=xinf1.copy()
    xinfm=xinf1.copy();xtaum=xinf1.copy()
    xinfh=xinf1.copy();xtauh=xinf1.copy()
    xinfj=xinf1.copy();xtauj=xinf1.copy()
    xinfd=xinf1.copy();xtaud=xinf1.copy()
    xinff=xinf1.copy();xtauf=xinf1.copy()
    xttab=xinf1.copy();x1=xinf1.copy()
    e1=xinf1.copy();ej=xinf1.copy()
    em=xinf1.copy();ed=xinf1.copy()
    eh=xinf1.copy();ef=xinf1.copy()
    v_values = xinf1.copy()
    #define physical parameters
    R = 8.3145  # J/(mol * °K) universal gas constant
    T = 273.15+37#°K physiologically normal body temperature 37°C
    F = 96485.3321233100184 # C/mol faraday's constant
    rtoverf=R*T/F
    # rtoverf=0.02650
    xk0=K_o# higher K_o should give shorter APD
    # gx1=0.282*2.837*np.sqrt(5.4/xk0)
    # gx1 modified to:
    gx1=0.423*2.837*np.sqrt(5.4/xk0)
    gk1=0.6047*np.sqrt(xk0/5.4)
    pr=0.01833
    xna0=140.
    xnai=18.
    xki=145.
    vx1=1000.*rtoverf*np.log((xk0+pr*xna0)/(xki+pr*xnai))
    # vx1=-87.94#in LuoRudy1990.pdf
    # vx1=-77.62#from wj's original table#### EK1 = -87.94mv in LuoRudy1990.pdf
    vk1=1000.*rtoverf*np.log(xk0/145.)
    v0=-100.
    v_values = np.arange(v0,v0+dv*(ndim),dv)
    for m in range(ndim-1):
        v=v_values[m]
        #compute values independent of gating variables using the functions defined postscript.
        xinf1[m]=a1(v)/(a1(v)+b1(v))
        xtau1[m]=1./(a1(v)+b1(v))
        xinfm[m]=am(v)/(am(v)+bm(v))
        xtaum[m]=1./(am(v)+bm(v))
        xinfh[m]=ah(v)/(ah(v)+bh(v))
        xtauh[m]=1./(ah(v)+bh(v))
        xinfj[m]=aj(v)/(aj(v)+bj(v))
        xtauj[m]=1./(aj(v)+bj(v))
        xinfd[m]=ad(v)/(ad(v)+bd(v))
        xtaud[m]=1./((ad(v)+bd(v)))
        xinff[m]=af(v)/(af(v)+bf(v))
        xtauf[m]=1./((af(v)+bf(v)))
        xttab[m]=xt(v,rtoverf=rtoverf,xk0=xk0)

        #precompute timesteps for gating variables
        fac=np.exp(0.04*(v+77.))
        fac1=(v+77.)*np.exp(0.04*(v+35.))
        if np.isclose(v,-77.,atol=1e-6):
            x1[m]=gx1*(v-vx1)*0.04/np.exp(0.04*(v+35.))
        else:
            x1[m]=gx1*(v-vx1)*(fac-1.)/fac1

        if (xtau1[m]<=5.e-4):
            e1[m]=0.
        else:
            e1[m]=np.exp(-dt/xtau1[m])

        if (xtauj[m]<=5.e-4):
            ej[m]=0.
        else:
            ej[m]=np.exp(-dt/xtauj[m])
        if (xtaud[m]<=5.e-4):
            ed[m]=0.
        else:
            ed[m]=np.exp(-dt/xtaud[m])
        if (xtauf[m]<=5.e-4):
            ef[m]=0.
        else:
            ef[m]=np.exp(-dt/xtauf[m])
        if (xtaum[m]<=5.e-4):
            em[m]=0.
        else:
            em[m]=np.exp(-dt/xtaum[m])
        if (xtauh[m]<=5.e-4):
            eh[m]=0.
        else:
            eh[m]=np.exp(-dt/xtauh[m])

    #return the table values as a tuple of numpy arrays
    arr10=np.stack((v_values,xinfh,xttab))
    arr11=np.stack((v_values,e1,em,ef))
    arr12=np.stack((v_values,ed,ej,eh))
    arr13=np.stack((v_values,xtaud,xtauf))
    arr39=np.stack((v_values,xinf1,xtau1,xinfm,xtaum,xinfh,xtauh,xinfj,xtauj,\
         xinfd,xtaud,xinff,xtauf,xttab,x1,e1,em,eh,ej,ed,ef))
    return (arr10,arr11,arr12,arr13,arr39)

################################
# helper functions
################################
def a1(v):
    cx1=.0005
    cx2=.083
    cx3=50.
    cx6=0.057
    cx7=1.
    a1=cx1*np.exp(cx2*(v+cx3))/(np.exp(cx6*(v+cx3))+cx7)
    return a1

def b1(v):
    dx1=.0013
    dx2=-.06
    dx3=20.
    dx6=-.04
    dx7=1.
    b1=dx1*np.exp(dx2*(v+dx3))/(np.exp(dx6*(v+dx3))+dx7)
    return b1

def am(v):
    cm3=47.13
    cm4=-0.32
    cm5=47.13
    cm6=-0.1
    cm7=-1.
    am=(cm4*(v+cm5))/(np.exp(cm6*(v+cm3))+cm7)
    return am

def bm(v):
    dm1=0.08
    dm2=-11.
    bm=dm1*np.exp(v/dm2)
    return bm

def ah(v):
    if (v>=-40.):
        ah=0.
    else:
        ch1=0.135
        ch2=-6.8
        ch3=80.
        ah=ch1*np.exp((v+ch3)/ch2)
    return ah

def bh(v):
    if (v>=-40.):
        dh1=0.13
        dh3=10.66
        dh6=-11.1
        dh7=1.
        bh=1./(dh1*(np.exp((v+dh3)/dh6)+dh7))
    else:
        dh1=3.56
        dh2=0.079
        dh3=310000.
        dh4=0.35
        bh=dh1*np.exp(dh2*v)+dh3*np.exp(dh4*v)
    return bh

def af(v):
    cf1=0.012
    cf2=-0.008
    cf3=28.
    cf6=0.15
    cf7=1.
    af=cf1*np.exp(cf2*(v+cf3))/(np.exp(cf6*(v+cf3))+cf7)
    return af

def bf(v):
    df1=0.0065
    df2=-.02
    df3=30.
    df6=-.2
    df7=1.
    bf=df1*np.exp(df2*(v+df3))/(np.exp(df6*(v+df3))+df7)
    return bf

def ad(v):
    cd1=0.095
    cd2=-0.01
    cd3=-5.
    cd6=-0.072
    cd7=1.
    ad=cd1*np.exp(cd2*(v+cd3))/(np.exp(cd6*(v+cd3))+cd7)
    return ad

def bd(v):
    dd1=0.07
    dd2=-.017
    dd3=44.
    dd6=0.05
    dd7=1.
    bd=dd1*np.exp(dd2*(v+dd3))/(np.exp(dd6*(v+dd3))+dd7)
    return bd

def aj(v):
    if (v>=-40.):
        aj=0.
    else:
        cj1=-127140.
        cj2=0.24444
        cj3=-0.00003474
        cj4=-0.04391
        cj5=37.78
        cj6=0.311
        cj7=79.23
        cj8=1.
        aj=(cj1*np.exp(cj2*v)+cj3*np.exp(cj4*v))*(v+cj5)\
                 /(np.exp(cj6*(v+cj7))+cj8)
    return aj

def bj(v):
    if (v>=-40.):
        dj1=0.3
        dj2=-0.0000002535
        dj3=32.
        dj6=-0.1
        dj7=1.
        bj=dj1*np.exp(dj2*v)/(np.exp(dj6*(v+dj3))+dj7)
    else:
        dj1=0.1212
        dj2=-0.01052
        dj3=40.14
        dj6=-0.1378
        dj7=1.
        bj=dj1*np.exp(dj2*v)/(np.exp(dj6*(v+dj3))+dj7)
    return bj

def xt(v,rtoverf,xk0,backcon=1.):
    '''total time independent potatassium current'''
    vk1=1000.*rtoverf*np.log(xk0/145.)
    vk1=-87.95
    gk1=0.6047*np.sqrt(xk0/5.4)
    ak1=1.02/(1.+np.exp(0.2385*(v-vk1-59.215)))
    bk1=(0.49124*np.exp(0.08032*(v-vk1+5.476))+np.exp(0.06175*\
        (v-vk1-594.31)))/(1.+np.exp(-0.5143*(v-vk1+4.753)))
    xk1=gk1*ak1*(v-vk1)/(ak1+bk1)
    xkp=0.0183*(v-vk1)/(1.+np.exp((7.488-v)/5.98))
    xbac=0.03921*(v+59.87)
    xt=xk1+xkp+backcon*xbac
    return xt

################################
# command line interface
################################
if __name__ == '__main__':
    import sys,os
    save_deserialized=True
    #suppose float arguments are a list of time step sizes
    dt_lst = [float(x) for x in sys.argv[1:]]
    if len(dt_lst)==0:
        raise(Exception("Error: zero float arguments given.  Enter a list of floats to serve as time step sizes!"))
    for dt in dt_lst:
        retval = generate_lookup_table_LR_I(dt=dt)
        #save serialized results for timestep dt
        if not os.path.exists('lookup_tables'):
            os.mkdir('lookup_tables')
        save_fn=f"lookup_tables/luo_rudy_dt_{dt}.npz"
        if save_deserialized:
            #save deserialized results for timestep dt
            arr10,arr11,arr12,arr13,arr39=retval
            fmt='%12.6f'#'%.18e'
            # np.savetxt(fname=save_fn.replace('.npz','_arr10.csv'),
            #                     X=arr10.T,fmt=fmt,delimiter=',')
            # np.savetxt(fname=save_fn.replace('.npz','_arr11.csv'),
            #                     X=arr11.T,fmt=fmt,delimiter=',')
            # np.savetxt(fname=save_fn.replace('.npz','_arr12.csv'),
            #                     X=arr12.T,fmt=fmt,delimiter=',')
            # np.savetxt(fname=save_fn.replace('.npz','_arr13.csv'),
                                # X=arr13.T,fmt=fmt,delimiter=',')
            np.savetxt(fname=save_fn.replace('.npz','_arr39.csv'),
                                X=arr39.T,fmt=fmt,delimiter=',')

        #(optional) save compressed/serialized arrays with self documenting keywords
        # np.savez_compressed(save_fn,*retval,
        #     kwds=[
        #         'arr10_v_xinfh_xttab',
        #         'arr11_v_e1_em_ef',
        #         'arr12_v_ed_ej_eh',
        #         'arr13_v_xtaud_xtauf',
        #         'arr39_v_xinf1_xtau1_xinfm_xtaum_xinfh_xtauh_xinfj_xtauj_xinfd_xtaud_xinff_xtauf_xttab_x1_e1_em_eh_ej_ed_ef'
        #         ])
