#!/usr/bin/env python3
# a humble worker for the PDE Simulation.
# Developer: Tim (the Tyrant) Tyree
# 3.31.2020
# each worker iteratively toils through its list of inputs in stdin.  
# for each input, a well-defined routine is performed 
# and the output returned to stdout.  In the case of the unexpected,
# it will report to stderr.

import pandas as pd
import numpy as np
from scipy.special import kn #modified bessel function of the second kind of integer order n
from scipy.interpolate import BSpline
# from joblib import Parallel, delayed
# from multiprocessing import cpu_count
# num_cores = cpu_count()# import multiprocessing# num_cores = multiprocessing.cpu_count()

#automate the boring stuff
import os, sys #, shutil,time
# beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()
# # print('notebook is at: ' + nb_dir)

###########################################################################
###### cell direction sensing #############################################
###########################################################################
def measure_at(c, loc, dr, thresh):
    '''returns 1, 0, -1 if cells would be moving away, neither, or towards the cell cluster, respectively.'''
    dface = np.hstack([ 0. , np.diff(c)/dr , 0. ])
    return sign_w_thresh(dface[loc], thresh = thresh)
def sign_w_thresh(arg,thresh):
	    '''returns 1 if arg is more than thresh, -1 if arg is less than -thresh, and zero else.
	    assumes thresh>0. See test cases for example usage.'''
	    if arg-thresh>0:
	        return 1
	    elif arg+thresh<0:
	        return -1
	    else:
	        return 0

#TODO: once I'm certin
def Hill(S,h,K):
	'''Eqn. 1 from Bader 2007. Relevant for PDE1 and PDE7.
	S = the concentration field of substrate, which is cAMP.
	h and K are floats'''
	return S**h/(K**h+S**h)

def get_Hill_degradation_term(c,pde,kPDE,b,h):
    '''let overall scale of degradation still vary as kPDE.'''
    boo = c>0
    out = 0*c
    out[boo] = -kPDE*np.divide(np.multiply(pde[boo],c[boo]**(h-1.)),(b**h+c[boo]**h))
    return out
    #(slow)     return -kPDE*np.divide(np.multiply(pde,c**(h-1.)),(b**h+c**h)) if c>0 else 0

def get_Hill_params(mode):
	if mode == "PDE1":
		return {
		# 'Vmax':95,#pmol/min per 10^7 cells ignore and work in units of kPDE
		'K':0.75,#\mu M
		'h':.8}
	if mode == "PDE7":
		return {
		'K':12.5,#\mu M
		'h':0.79}
	if mode == "PDE4":
		return {
		'K':9.6,#\mu M
		'h':1}
	return None

###########################################################################
###### make time dependent source from Martiel-Goldbeter (1987) ###########
###########################################################################
#if mg_scaled.csv is in the notebook's directory
workspace_dir = os.getcwd()
source_fn = 'mg_source_1987.csv'
assert(os.path.exists(source_fn))
#updated
def import_mg_source(period, mg_scaled_fn='mg_source_1987.csv', nn = 3, ):
    '''period is in minutes'''
    df = pd.read_csv(mg_scaled_fn)
    t_values = df.times.values
    a_values = df.synth_rate.values
    spl = BSpline(t_values, a_values, nn, extrapolate='True')
    Dt= 0.0005#minutes
    t_list = np.arange(0,11,Dt)#minutes
    c_list = spl(t_list)
    t_list_2 = np.arange(11,period+Dt,Dt)
    c_list_2 = 0*t_list_2#append zeros so vector is of period duration
    #assumption that spl(11)~1e-5 is sufficiently close to zero
    t_list = np.concatenate([t_list,t_list_2])*60#seconds as argument
    c_list = np.concatenate([c_list,c_list_2])
    spl2   = BSpline(t_list, c_list, nn, extrapolate='periodic')
    return spl2

def evaluate_mg_source(spl, t_list):
	'''returns cAMP current from cluster evaluated at the relevant time points.'''
	dsource = dict(zip(t_list,spl(t_list)))
	return dsource

###########################################################################
###### define time step  ##################################################
###########################################################################
def time_step(c, pde, rmesh, D, kPDE, dr, degrade, fluxLeft, fluxRight=0):
	'''returns rate of change of the field c defined over the 1D radial mesh, rmesh, 
	explicitely using only current state.'''
	#step one: update exterior faces and other state variables 
	dface = np.hstack([ 0. , np.diff(c) , 0. ])
	cp    = np.hstack([c[0]-dface[0], c, c[-1]+dface[-1]])
	face  = cp[1:]*0.5 + cp[:-1]*0.5

	#step two: calculate transient term 
	term1 = D*np.diff(face)/rmesh/dr
	term2 = D*np.diff(dface)/dr**2
	termDEG = degrade
	
	#the following is the original mode 0
	# termDEG = -1*kPDE*c*pde

	#add flux term
	termBC = np.hstack([ fluxLeft , 0*c[1:-1] , fluxRight ])

	#step three: integrate in time
	dcdt  = term1 + term2 + termBC + termDEG
	return dcdt

###########################################################################
###### define simulation ###########
###########################################################################
def simulate(kPDE, LPDE, c0, T, iter_no, dt, time_res, mode):
	'''function that returns a minimalist df.  
	T = period in minutes.
	dt = step size in seconds.'''
	#initialize
	#  make the mesh
	r0   = 50. #um
	Lr   = 1000.#um
	dr   = 1.  #um
	Ts   = np.multiply(T,60.)
	#period of signal in seconds
	rmesh= np.arange(r0,Lr+dr,dr)
	c    = 0.*rmesh
	#  define default parameters
	F0   = 10**4
	D    = 100.#um^2/s
	ds   =(0.1/15)

	#  precompute realistic periodic camp signaling
	t_list  = np.around(np.arange(0,Ts+dt,dt),4)
	spl     = import_mg_source(period=T)
	dsource = evaluate_mg_source(spl, t_list)
	phi     = lambda t: F0*dsource[np.around(t,4)]/rmesh[0]

	hill_params = get_Hill_params(mode=mode)
	h = hill_params['h']
	b = hill_params['K']

	# initialize with numpy signatures to speed up to about twice the previous run time (original mode==0 case) 
	mesh_size = len(rmesh)
	degrade = np.vectorize(lambda c,pde:get_Hill_degradation_term(c,pde,kPDE=kPDE,b=b,h=h),
		signature = f"({mesh_size}),({mesh_size})->({mesh_size})")

	#cylindrical FEM with localized PDE decay
	# define b.c.'s in units of slope #concentration difference per time step
	fluxLeft = lambda t:phi(t%Ts)
	fluxRight= 0./rmesh[-1]
	
	#initialize field values
	c = c0+0.*rmesh
	pde = kn(0,rmesh/LPDE)/kn(0,r0/LPDE)#constitutive pde production in 2D radial coords

	#initialize loop invariants. iter_no>cycle_no. T>time>=t2.
	cycle_no = 0
	time = 0.
	t2 = time
	r_bins = [100,200,300,400,500,800] #bin location in microns
	rj_bins= [int((r-r0)/dr) for r in r_bins]#bin id
	v_bucket = [[] for r in r_bins]

	#print inputs
	print(f"Printing Inputs:")
	print(f"kPDE:{kPDE} LPDE:{LPDE} c0:{c0} T:{T} iter_no:{iter_no} dt:{dt} time_res:{time_res} mode:{mode}")

	#print headings for output data 
	print(f"\nPrinting Outputs including mean cell direction (mcd):")
	print('cycle_no,mean_c,mcd_at_{0:d},mcd_at_{1:d},mcd_at_{2:d},mcd_at_{3:d},mcd_at_{4:d},mcd_at_{5:d}'.format(*r_bins))
	# print(f'cycle_no, {r_bins[0]:3.3f}, {r_bins[1]:3.3f}, {r_bins[2]:3.3f}')
	#numerically integrate
	while(cycle_no < iter_no):
		#step forward in time (FEM)
		dcdt = time_step(c, pde, rmesh, D, kPDE, dr,  degrade=degrade(c,pde), fluxLeft=fluxLeft(time))
		c = c + dt*dcdt

		#TODO(later): play with runge-kutta stability at larger time steps.
		#(not working):step forward in time using iterative implicit euler method)
		# dcdt2 = time_step(c+dt/2*dcdt, pde, rmesh, D, kPDE, dr, fluxLeft(t2))
		# dcdt3 = time_step(c+dt/2*dcdt2, pde, rmesh, D, kPDE, dr, fluxLeft(t2))
		# c = c + dt*dcdt3
		#TODO(later): try iterating time_step a couple times for stability

		#every time_res seconds, measure direction of cell motion
		if t2>=time_res:
			#calculate direction of cell motion for each bin
			v_bins = [measure_at(c,r,dr=dr, thresh=ds) for r in rj_bins]
			#record cell direction
			for i,l in enumerate(v_bucket):
			    l.append(v_bins[i])
			#reset t2
			t2 = 0
		#at the end of every period, print average cell motion
		if time > Ts:
			#calculate mean cell motion for each bin
			v_mean = [float(sum(v)/len(v)) for v in v_bucket]
			#calculate instantaneous mean camp field outside of the cluster
			c_mean = 2*dr*np.dot(c,rmesh)/(Lr**2-r0**2)

			#print cycle number followed by average cell motion for each bin
			print(f"{cycle_no},{c_mean:.5f},{v_mean[0]:.5f},{v_mean[1]:.5f},{v_mean[2]:.5f},{v_mean[3]:.5f},{v_mean[4]:.5f},{v_mean[5]:.5f}")
			#reset time, increment cycle_no
			time = 0
			cycle_no += 1

		#move time forward
		time += dt
		t2   += dt
	return True

##################################################
## main routine - run the simulation for some batch of parameters
##################################################   
if __name__ == "__main__":
	# parse arguments
	kPDE = float(sys.argv[1].split(',')[0])
	LPDE = float(sys.argv[2].split(',')[0])
	c0   = float(sys.argv[3].split(',')[0])
	T    = int(float(sys.argv[4].split(',')[0])) 
	iter_no  = int(float(sys.argv[5].split(',')[0]))
	dt       = float(sys.argv[6].split(',')[0])
	time_res = int(float(sys.argv[7].split(',')[0]))
	mode = 'PDE'+str(int(float(sys.argv[8].split(',')[0])))
	retval = simulate(kPDE=kPDE, LPDE=LPDE, c0=c0, T=T, iter_no=iter_no, dt=dt, time_res = time_res, mode=mode)
	#yes, ^this type casting should be necessary.
	#final argument is '1', '4', or '7' for PDE1,4,or7, respectively.  Keep the datatypes small	