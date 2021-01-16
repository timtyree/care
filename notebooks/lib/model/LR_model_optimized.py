#!/bin/bash/env python3
#Timothy Tyree
#1.13.2021
# The model from Luo-Rudy (1990) modified as described 
# in Qu et alli (2000) to exhibit spiral defect chaos
# implemented on a square computational domain.
# Uses lookup table instead of using comp_ionic_currents
from numba import njit, jit
import numpy as np
import os

@njit
def comp_transient_gating_variable(var, tau, varinfty):
	return (varinfty - var)/tau


def get_comp_dVcdt(width=200,height=200,diffCoef=0.001,ds=5.,Cm=1.):
	'''the square domain has area = dsdpixel**2*width*height
	dsdpixel=0.025 cm/pixel by default
	'''
	comp_ionic_flux=get_comp_ionic_flux()
	#spatial discretization
	cddx = width  / ds  #if this is too big than the simulation will blow up (at a given timestep)
	cddy = height / ds #if this is too big than the simulation will blow up (at a given timestep)
	cddx *= cddx
	cddy *= cddy
	@njit
	def comp_dVcdt(inVc,x,y,inCgate,IK1T,x1):
		V,Ca_i = inVc[x,y].T
		# Compute Ionic Current Density
		Iion, dCa_i_dt=comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i)
		# Compute transient term for transmembrane voltage
		dVltdt  = laplacian(inVc, x, y, cddx, cddy, V)
		dVltdt *= float(diffCoef)
		dVltdt -= float(Iion/Cm)
		dVcdt_val=np.array([dVltdt,dCa_i_dt],dtype=np.float64)
		return dVcdt_val
	return comp_dVcdt

def get_arr39(dt,nb_dir):
	cwd=os.getcwd()
	#generate lookup tables for timestep
	os.chdir(os.path.join(nb_dir,'lib/model'))
	cmd=f"python3 gener_table.py {dt}"
	os.system(cmd)
	#load lookup table for constant timestep, dt.
	os.chdir(os.path.join(nb_dir,'lib/model','lookup_tables'))
	table_fn=f"luo_rudy_dt_{dt}.npz"
	table_data=np.load(table_fn)    
	#convert table_data to a numpy array
	kwds=table_data.get('kwds')
	cols=kwds[-1].split('_')[1:]
	keys=list(table_data.keys())
	arr39=table_data[keys[-1]].T  
	#return to original working directory
	os.chdir(cwd)
	return arr39

@njit
def comp_exact_next_gating_var(inCgate,outCgate,arr_interp):
	"""returns updated  gating variables"""
	m = inCgate[0] #activation gate parameter (Na)
	h = inCgate[1] #fast inactivation gate parameter (INa)
	j = inCgate[2] #slow inactivation gate parameter (INa)
	d = inCgate[3] #activation gate parameter (Isi)
	f = inCgate[4] #inactivation gate parameter (Isi)
	x_var = inCgate[5] #activation gate parameter (IK)

	#parse the linearly interpolated row
	x_infty =arr_interp[1]    # 'xinf1', 
	# tau_x =arr_interp[2]    # 'xtau1', 
	m_infty=arr_interp[3]    # 'xinfm', 
	# tau_m=arr_interp[4]    # 'xtaum', 
	h_infty=arr_interp[5]    # 'xinfh', 
	# tau_h=arr_interp[6]    # 'xtauh', 
	j_infty=arr_interp[7]    # 'xinfj',
	# tau_j=arr_interp[8]    # 'xtauj', 
	d_infty=arr_interp[9]    # 'xinfd', 
	# tau_d=arr_interp[10]    # 'xtaud', 
	f_infty=arr_interp[11]    # 'xinff', 
	# tau_f=arr_interp[12]    # 'xtauf', 
	#     IK1T=arr_interp[13]    # 'xttab', 
	x1=arr_interp[14]    # 'x1', 
	e1=arr_interp[15]    # 'e1',
	em=arr_interp[16]    # 'em', 
	eh=arr_interp[17]    # 'eh', 
	ej=arr_interp[18]    # 'ej',
	ed=arr_interp[19]    # 'ed', 
	ef=arr_interp[20]    # 'ef'

	outCgate[0]=comp_soln_gating_var(m,m_infty, em)
	outCgate[1]=comp_soln_gating_var(h,h_infty, eh)
	outCgate[2]=comp_soln_gating_var(j,j_infty, ej)
	outCgate[3]=comp_soln_gating_var(d,d_infty, ed)
	outCgate[4]=comp_soln_gating_var(f,f_infty, ef)
	outCgate[5]=comp_soln_gating_var(x_var,x_infty, e1)  

@njit
def comp_curr_transient_gating_var(inCgate,outCgate,arr_interp):
	"""returns updated  gating variables"""
	m = inCgate[0] #activation gate parameter (Na)
	h = inCgate[1] #fast inactivation gate parameter (INa)
	j = inCgate[2] #slow inactivation gate parameter (INa)
	d = inCgate[3] #activation gate parameter (Isi)
	f = inCgate[4] #inactivation gate parameter (Isi)
	x_var = inCgate[5] #activation gate parameter (IK)

	#parse the linearly interpolated row
	x_infty =arr_interp[1]    # 'xinf1', 
	tau_x =arr_interp[2]    # 'xtau1', 
	m_infty=arr_interp[3]    # 'xinfm', 
	tau_m=arr_interp[4]    # 'xtaum', 
	h_infty=arr_interp[5]    # 'xinfh', 
	tau_h=arr_interp[6]    # 'xtauh', 
	j_infty=arr_interp[7]    # 'xinfj',
	tau_j=arr_interp[8]    # 'xtauj', 
	d_infty=arr_interp[9]    # 'xinfd', 
	tau_d=arr_interp[10]    # 'xtaud', 
	f_infty=arr_interp[11]    # 'xinff', 
	tau_f=arr_interp[12]    # 'xtauf', 
	# IK1T=arr_interp[13]    # 'xttab', 
	# x1=arr_interp[14]    # 'x1', 
	# e1=arr_interp[15]    # 'e1',
	# em=arr_interp[16]    # 'em', 
	# eh=arr_interp[17]    # 'eh', 
	# ej=arr_interp[18]    # 'ej',
	# ed=arr_interp[19]    # 'ed', 
	# ef=arr_interp[20]    # 'ef'

	outCgate[0]=comp_transient_gating_variable(m,tau_m,m_infty)
	outCgate[1]=comp_transient_gating_variable(h,tau_h,h_infty)
	outCgate[2]=comp_transient_gating_variable(j,tau_j,j_infty)
	outCgate[3]=comp_transient_gating_variable(d,tau_d,d_infty)
	outCgate[4]=comp_transient_gating_variable(f,tau_f,f_infty)
	outCgate[5]=comp_transient_gating_variable(x_var,tau_x,x_infty)
	
def get_comp_ionic_flux(GNa=16.,GK1=0.6047,Gsi=0.052,EK1=-87.94,Eb=-59.87,ENa=54.4,GK=0.423):
	# #maximum conductances
	# GNa = 16.     #mS/cm^2 from Qu2000.pdf #GNa=23 in Luo1990.pdf
	# GK1 = 0.6047  #mS/cm^2 from Qu2000.pdf
	# Gsi = 0.052   #mS/cm^2 spiral wave breakup phase from Qu2000.pdf
	# GK  = 0.423   #mS/cm^2 #from Qu2000.pdf
	# #reversal potentials
	# EK1 = -87.94 #mV
	# EKp = EK1    #mV
	# Eb  = -59.87 #mV
	# ENa = 54.4   #mV
	@njit
	def comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i):
		"""returns updated  gating variables"""
		m = inCgate[0] #activation gate parameter (Na)
		h = inCgate[1] #fast inactivation gate parameter (INa)
		j = inCgate[2] #slow inactivation gate parameter (INa)
		d = inCgate[3] #activation gate parameter (Isi)
		f = inCgate[4] #inactivation gate parameter (Isi)
		x_var = inCgate[5] #activation gate parameter (IK)
		#Fast sodium current
		INa = GNa*m**3*h*j*(V-ENa)
		#Slow inward current
		Esi=7.7-13.0287*np.log(Ca_i)#mV
		Isi=Gsi*d*f*(V-Esi)
		#time dependent potassium current 
		IK=x_var*x1#GK*x_var*x1#
		#total electric current
		Iion=INa+IK1T+Isi+IK
		#calcium uptake rate (dominated by activity of the sarcoplasmic reticulum)
		dCa_i_dt=-10**-4*Isi+0.07*(10**-4-Ca_i)
		return Iion, dCa_i_dt#INa,IK1T,Isi,IK# GNa,m**3*h*j,(V-ENa)#,#
	return comp_ionic_flux

@njit
def comp_soln_gating_var(var, varinfty, evar):
	return varinfty - (varinfty-var)*evar

def get_comp_v_row(v_values):
	dv=np.around(np.mean(np.diff(v_values)),1)
	v0=np.min(v_values)
	@njit
	def comp_v_row(v):
		return int((v-v0)/dv)
	return comp_v_row

def get_lookup_params(v_values,dv=0.1):
	comp_row=get_comp_v_row(v_values)
	@njit
	def lookup_params(V,arr39):
		M=comp_row(V)
		arr=arr39[M:M+2]
		Vlo=arr[0,0]
		frac=(V-Vlo)/dv
		arr_interp=frac*arr[1,:]+(1.-frac)*arr[0,:]
		# assert ( np.isclose( arr_interp[0]-V, 0.))#passed
		return arr_interp
	return lookup_params

# /*------------------------------------------------------------------------
#  * periodic boundary conditions for each read from textures
#  *------------------------------------------------------------------------
#  */
@njit
def pbc(S,x,y):
	'''S=texture with size 512,512,3
	(x, y) pixel coordinates of texture with values 0 to 1.
	tight boundary rounding is in use.'''
	width  = int(S.shape[0])
	height = int(S.shape[1])
	if ( x < 0  ):				# // Left P.B.C.
		x = width - 1
	elif ( x > (width - 1) ):	# // Right P.B.C.
		x = 0
	if( y < 0 ):				# //  Bottom P.B.C.
		y = height - 1
	elif ( y > (height - 1)):	# // Top P.B.C.
		y = 0
	return S[x,y]

# /*-------------------------------------------------------------------------
#  * Laplacian
#  *-------------------------------------------------------------------------
#  */ 

@njit
def laplacian(inVfs, x, y, cddx, cddy, V):
	#five point stencil
	dVltdt = (
	    (pbc(inVfs, x + 1, y)[0] - 2.0 * V +
	     pbc(inVfs, x - 1, y)[0]) * cddx +
	    (pbc(inVfs, x, y + 1)[0] - 2.0 * V +
	     pbc(inVfs, x, y - 1)[0]) * cddy)
	return dVltdt
#(deprecated) nine point stencil
# 	dVlt2dt = (1. - 1. / 3.) * (
# 		(pbc(inVfs, x + 1, y)[0] - 2.0 * C[0] +
# 		 pbc(inVfs, x - 1, y)[0]) * cddx +
# 		(pbc(inVfs, x, y + 1)[0] - 2.0 * C[0] +
# 		 pbc(inVfs, x, y - 1)[0]) * cddy) + (1. / 3.) * 0.5 * (
# 			 pbc(inVfs, x + 1, y + 1)[0] + pbc(
# 				 inVfs, x + 1, y - 1)[0] + pbc(inVfs, x - 1, y - 1)[0] +
# 			 pbc(inVfs, x - 1, y + 1)[0] - 4.0 * C[0]) * (cddx + cddy)

# /*-------------------------------------------------------------------------
#  * Adaptive time steping
#  *-------------------------------------------------------------------------
#  */ 

@njit
def comp_next_time_step(dt_prev, dV):
	''' returns the size of the next time step
	Adaptive time stepping as described in Luo1990.pdf.
	During the stimulus, a fixed time step (0.05 or 0.01 msec) 
	should be used to minimize variability in the stimulus duration 
	caused by the time discretizationprocedure.
	'''
	dVmax = 0.8 #mV
	dVmin = 0.2 #mV
	dtmin = 0.01#ms
	dtmax = 0.3 #ms
	Vdot = dV/dt_prev
	if dV<=dVmin:
		#for relatively slow changes in voltage
		dt_next = dVmax/Vdot
	elif dV>=dVmax:
		#for relatively fast changes in voltage
		dt_next = dVmin/Vdot
	else:
		#TODO: if this dt_next results in dV>=dVmax, dt should be reduced until the complimentary condition dV<dVmax is met.
		dt_next = dt_prev/2.
	return dt_next