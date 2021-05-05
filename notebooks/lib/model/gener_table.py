#!/usr/bin/env python3
# This program determines the look-up table for the LR-I model
# to generate lookup tables for a time step of dt=0.01 ms from command line, execute
# $ python3 gener_table.py 0.01
from numpy import sqrt, exp, log
import numpy as np

def program_br(ndimtab=2500,dt=0.001,dv=0.1,xspeed=1.,backon=1.,
    K_o=5.4#50#3.8#5.4#7#
    ,dtype=np.float64, order='F'):
    # implicit real*8(a-h,o-z)
    # common/par/backcon,xk0,rtoverf
    # parameter (ndimtab=2500)
    #preallocate memory
    xinf1=np.zeros(ndimtab, dtype=dtype, order=order)#enforces fortran ordering, 64 bit floating precision
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
    # write(6,*)'input xspeed,backcon,  dt'
    # read(5,*)xspeed,backcon, dt
    #other parameters
    R = 8.3145  # J/(mol * °K) universal gas constant
    T = 273.15+37#°K physiologically normal body temperature 37°C
    F = 96485.3321233100184 # C/mol faraday's constant
    # R=8.315
    # T=310.15
    # F=96490.
    rtoverffull=R*T/F
    rtoverf=rtoverffull#0.02650
    # write(6,*)rtoverffull,rtoverf
    xk0=K_o#,5.4#7. higher K_o should give shorter APD#
    # c     gx1=0.282*2.837*sqrt(5.4/xk0)
    # c modified to:
    gx1=0.423*2.837*sqrt(5.4/xk0)
    gk1=0.6047*sqrt(xk0/5.4)
    pr=0.01833
    xna0=140.
    xnai=18.
    xki=145.
    vx1=1000.*rtoverf*log((xk0+pr*xna0)/(xki+pr*xnai))
    # vx1=-87.94#in LuoRudy1990.pdf
    # vx1=-77.62#from wj's original table#### EK1 = -87.94mv in LuoRudy1990.pdf
    vk1=1000.*rtoverf*log(xk0/145.)
    # write(6,*)'xk0,gx1,gk1,vx1,vk1'
    # write(6,*)xk0,gx1,gk1,vx1,vk1
    v0=-100.
    v_values = np.arange(v0,v0+dv*(ndimtab),dv)
    for m in range(ndimtab-1):
        # v=np.around(v0+m*dv,decimals=2)#fixes numerical error from floating point arithmetic
        v=v_values[m]
        xinf1[m]=a1(v)/(a1(v)+b1(v))
        xtau1[m]=1./(a1(v)+b1(v))
        xinfm[m]=am(v)/(am(v)+bm(v))
        xtaum[m]=1./(am(v)+bm(v))
        xinfh[m]=ah(v)/(ah(v)+bh(v))
        xtauh[m]=1./(ah(v)+bh(v))
        xinfj[m]=aj(v)/(aj(v)+bj(v))
        xtauj[m]=1./(aj(v)+bj(v))
        xinfd[m]=ad(v)/(ad(v)+bd(v))
        # c divide both d and f channels time const. by xspeed
        xtaud[m]=1./((ad(v)+bd(v))*xspeed)
        xinff[m]=af(v)/(af(v)+bf(v))
        xtauf[m]=1./((af(v)+bf(v))*xspeed)
        xttab[m]=xt(v,rtoverf=rtoverf,xk0=xk0)

        fac=exp(0.04*(v+77.))
        fac1=(v+77.)*exp(0.04*(v+35.))
        if np.isclose(v,-77.,atol=1e-6):# then
            x1[m]=gx1*(v-vx1)*0.04/exp(0.04*(v+35.))
        else:
            x1[m]=gx1*(v-vx1)*(fac-1.)/fac1

        if (xtau1[m]<=5.e-4):# then
            e1[m]=0.
        else:
            e1[m]=exp(-dt/xtau1[m])

        if (xtauj[m]<=5.e-4):# then
            ej[m]=0.
        else:
            ej[m]=exp(-dt/xtauj[m])
        if (xtaud[m]<=5.e-4):# then
            ed[m]=0.
        else:
            ed[m]=exp(-dt/xtaud[m])
        if (xtauf[m]<=5.e-4):# then
            ef[m]=0.
        else:
            ef[m]=exp(-dt/xtauf[m])
        if (xtaum[m]<=5.e-4):# then
            em[m]=0.
        else:
            em[m]=exp(-dt/xtaum[m])
        if (xtauh[m]<=5.e-4):# then
            eh[m]=0.
        else:
            eh[m]=exp(-dt/xtauh[m])

        # write(10,*)v,xinfh[m],xttab[m]
        # write(11,*)v,e1[m],em[m],ef[m]
        # write(12,*)v,ed[m],ej[m],eh[m]
        # write(13,*)v,xtaud[m],xtauf[m]
    # enddo
    # open(unit=39,file='table_qu',status='new')
    # write(39)xspeed,backcon
    # write(39)xinf1,xtau1,xinfm,xtaum,xinfh,xtauh,xinfj,xtauj,
    # 1     xinfd,xtaud,xinff,xtauf,xttab,x1,e1,em,eh,ej,ed,ef
    # close(39)
    # stop
    # 9     format(10(f12.6,1x))

    #return the table values as a tuple of numpy arrays
    arr10=np.stack((v_values,xinfh,xttab))
    arr11=np.stack((v_values,e1,em,ef))
    arr12=np.stack((v_values,ed,ej,eh))
    arr13=np.stack((v_values,xtaud,xtauf))
    arr39=np.stack((v_values,xinf1,xtau1,xinfm,xtaum,xinfh,xtauh,xinfj,xtauj,\
         xinfd,xtaud,xinff,xtauf,xttab,x1,e1,em,eh,ej,ed,ef))
    return (arr10,arr11,arr12,arr13,arr39)

def a1(v):
    # implicit real*8(a-h,o-z)
    cx1=.0005
    cx2=.083
    cx3=50.
    cx6=0.057
    cx7=1.
    a1=cx1*exp(cx2*(v+cx3))/(exp(cx6*(v+cx3))+cx7)
    return a1

def b1(v):
    # implicit real*8(a-h,o-z)
    dx1=.0013
    dx2=-.06
    dx3=20.
    dx6=-.04
    dx7=1.
    b1=dx1*exp(dx2*(v+dx3))/(exp(dx6*(v+dx3))+dx7)
    return b1

def am(v):
    # implicit real*8(a-h,o-z)
    cm3=47.13
    cm4=-0.32
    cm5=47.13
    cm6=-0.1
    cm7=-1.
    am=(cm4*(v+cm5))/(exp(cm6*(v+cm3))+cm7)
    return am

def bm(v):
    # implicit real*8(a-h,o-z)
    dm1=0.08
    dm2=-11.
    bm=dm1*exp(v/dm2)
    return bm

def ah(v):
    # implicit real*8(a-h,o-z)
    if (v>=-40.):# then
        ah=0.
    else:
        ch1=0.135
        ch2=-6.8
        ch3=80.
        ah=ch1*exp((v+ch3)/ch2)
    return ah

def bh(v):
    # implicit real*8(a-h,o-z)
    if (v>=-40.):# then
        dh1=0.13
        dh3=10.66
        dh6=-11.1
        dh7=1.
        bh=1./(dh1*(exp((v+dh3)/dh6)+dh7))
    else:
        dh1=3.56
        dh2=0.079
        dh3=310000.#3.1d5
        dh4=0.35
        bh=dh1*exp(dh2*v)+dh3*exp(dh4*v)
    return bh

def af(v):
    # implicit real*8(a-h,o-z)
    cf1=0.012
    cf2=-0.008
    cf3=28.
    cf6=0.15
    cf7=1.
    af=cf1*exp(cf2*(v+cf3))/(exp(cf6*(v+cf3))+cf7)
    return af

def bf(v):
    # implicit real*8(a-h,o-z)
    df1=0.0065
    df2=-.02
    df3=30.
    df6=-.2
    df7=1.
    bf=df1*exp(df2*(v+df3))/(exp(df6*(v+df3))+df7)
    return bf

def ad(v):
    # implicit real*8(a-h,o-z)
    cd1=0.095
    cd2=-0.01
    cd3=-5.
    cd6=-0.072
    cd7=1.
    ad=cd1*exp(cd2*(v+cd3))/(exp(cd6*(v+cd3))+cd7)
    return ad

def bd(v):
    # implicit real*8(a-h,o-z)
    dd1=0.07
    dd2=-.017
    dd3=44.
    dd6=0.05
    dd7=1.
    bd=dd1*exp(dd2*(v+dd3))/(exp(dd6*(v+dd3))+dd7)
    return bd

def aj(v):
    # implicit real*8(a-h,o-z)
    if (v>=-40.):# then
        aj=0.
    else:
        cj1=-127140.#-1.2714d5
        cj2=0.24444
        cj3=-0.00003474#-3.474d-5
        cj4=-0.04391
        cj5=37.78
        cj6=0.311
        cj7=79.23
        cj8=1.
        aj=(cj1*exp(cj2*v)+cj3*exp(cj4*v))*(v+cj5)\
                 /(exp(cj6*(v+cj7))+cj8)
    return aj

def bj(v):
    # implicit real*8(a-h,o-z)
    if (v>=-40.):# then
        dj1=0.3
        dj2=-0.0000002535#-2.535d-7
        dj3=32.
        dj6=-0.1
        dj7=1.
        bj=dj1*exp(dj2*v)/(exp(dj6*(v+dj3))+dj7)
    else:
        dj1=0.1212
        dj2=-0.01052
        dj3=40.14
        dj6=-0.1378
        dj7=1.
        bj=dj1*exp(dj2*v)/(exp(dj6*(v+dj3))+dj7)
    return bj

def xt(v,rtoverf,xk0,backcon=1.):
    # implicit real*8(a-h,o-z)
    # common/par/backcon,xk0,rtoverf #total time independent potatassium current
    vk1=1000.*rtoverf*log(xk0/145.) # xk0=5.4
    vk1=-87.95
    gk1=0.6047*sqrt(xk0/5.4)
    ak1=1.02/(1.+exp(0.2385*(v-vk1-59.215)))
    bk1=(0.49124*exp(0.08032*(v-vk1+5.476))+exp(0.06175*\
        (v-vk1-594.31)))/(1.+exp(-0.5143*(v-vk1+4.753)))
    xk1=gk1*ak1*(v-vk1)/(ak1+bk1)
    xkp=0.0183*(v-vk1)/(1.+exp((7.488-v)/5.98))
    xbac=0.03921*(v+59.87)
    # c MULTIPLY xbac with backcon
    xt=xk1+xkp+backcon*xbac
    return xt

#command line interface
if __name__ == '__main__':
    import sys,os
    save_deserialized=True
    #suppose float arguments are a list of time step sizes
    dt_lst = [float(x) for x in sys.argv[1:]]
    if len(dt_lst)==0:
        raise(Exception("Error: zero float arguments given"))
    for dt in dt_lst:
        retval = program_br(dt=dt)
        #save serialized results for timestep dt
        if not os.path.exists('lookup_tables'):
            os.mkdir('lookup_tables')
        save_fn=f"lookup_tables/luo_rudy_dt_{dt}.npz"
        #save compressed arrays with self documenting keywords
        # np.savez_compressed(save_fn,*retval,
        #     kwds=[
        #         'arr10_v_xinfh_xttab',
        #         'arr11_v_e1_em_ef',
        #         'arr12_v_ed_ej_eh',
        #         'arr13_v_xtaud_xtauf',
        #         'arr39_v_xinf1_xtau1_xinfm_xtaum_xinfh_xtauh_xinfj_xtauj_xinfd_xtaud_xinff_xtauf_xttab_x1_e1_em_eh_ej_ed_ef'
        #         ])
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
            # print(arr10.shape)
            # print(arr11.shape)
            # print(arr12.shape)
            # print(arr13.shape)
            # print(arr39.shape)
