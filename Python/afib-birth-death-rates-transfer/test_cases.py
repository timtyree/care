#!/usr/bin/env python
#test cases for pde-sim

from pde_sim_worker import *
import pde_sim_worker_hill as hill
from pde_sim_grid_scan import *
import numpy as np
import pandas as pd


#initialize
#  make the mesh
r0   = 50. #um
Lr   = 1000.#um
dr   = 1.  #um
dt   = 0.0005#sec 
rmesh= np.arange(r0,Lr+dr,dr)
c    = 0.*rmesh
#  define default parameters
F0   = 10**4
D    = 100.#um^2/s
ds   = 0.1/15#nM/µm


#test case for grid_to_df
assert(grid_to_df(
    kgrid=[0.02,0.002],
    Lgrid=[100.0],
    c0grid=[4],
    Tgrid=[21,20],
    itnogrid=[2],
    dtgrid=[0.002],
    trgrid=[10],
).size==28)
assert(grid_to_df(
    kgrid=[0.02,0.002],
    Lgrid=[100.0],
    c0grid=[4],
    Tgrid=[21,20,21],
    itnogrid=[2],
    dtgrid=[0.002],
    trgrid=[10],
).size==42)

#test case for cell direction measurement
pde = kn(0,rmesh/100)/kn(0,r0/100)#constitutive pde production in 2D radial coords
assert(measure_at(pde,10, dr=dr, thresh=ds)==-1)
assert(measure_at(pde,300, dr=dr, thresh=ds)==0)

#TODO:test case for mg source
#test functional value retrieval has the right dtype and value
#test T*60 periodic for T = 21

#TODO:test case for functional simulate call


#test case for hill degradation term
h=0.8; c = np.arange(50,1000,1); b = 0.75
assert(len(list(hill.get_Hill_degradation_term(1,c,1,1,c)))==len(list(c)))