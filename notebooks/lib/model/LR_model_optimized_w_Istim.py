#!/bin/bash/env python3
#Timothy Tyree
#1.13.2021
# The model from Luo-Rudy (1990) modified as described
# in Qu et alli (2000) to exhibit spiral defect chaos
# implemented on a square computational domain.
# Uses lookup table instead of using comp_ionic_currents
from numba import njit, jit, float64#, prange
import numpy as np, pandas as pd
import os
from math import log

def get_one_step_map(nb_dir,dt,dsdpixel,width,height,**kwargs):
	'''Example Usage:
	dt, one_step_map= get_one_step_map(nb_dir,dt,dsdpixel,width,height,**kwargs):
'''
	#make null stimulus
	ds=dsdpixel*width
	# txt_Istim_none=np.zeros(shape=(width,height,1), dtype=np.float64, order='C')
	dt, kernelA, kernelB=get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds)

	#get one step map
	# txt_Istim=txt_Istim_none.copy()
	@njit
	def one_step_map(txt,txt_Istim):
		kernelA(txt,txt_Istim)
		kernelB(txt,txt_Istim)

	return dt, one_step_map

@njit
def comp_transient_gating_variable(var, tau, varinfty):
	return (varinfty - var)/tau

def get_comp_dVcdt(width,height,diffCoef=0.001,ds=5.,Cm=1., **kwargs):
	'''the square domain has area = dsdpixel**2*width*height
	dsdpixel=0.025 cm/pixel by default
	'''
	comp_ionic_flux=get_comp_ionic_flux()
	#spatial discretization
	cddx = width  / ds  #if this is too big than the simulation will blow up (at a given timestep)
	cddy = height / ds #if this is too big than the simulation will blow up (at a given timestep)
	cddx *= cddx
	cddy *= cddy
	laplacian=get__laplacian(width,height,cddx,cddy)
	@njit
	def comp_dVcdt(inVc,x,y,inCgate,IK1T,x1):
		'''Example Usage:
		dVcdt_val=comp_dVcdt(inVc,x,y,inCgate,IK1T,x1)
		'''
		V,Ca_i = inVc[x,y].T
		# Compute Ionic Current Density
		Iion, dCa_i_dt=comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i)
		# Compute transient term for transmembrane voltage
		dVltdt  = laplacian(inVc, x, y, cddx, cddy, V)
		dVltdt *= float(diffCoef)
		dVltdt -= float(Iion/Cm)
		dVcdt_val=np.array([dVltdt,dCa_i_dt],dtype=float64)
		return dVcdt_val
	return comp_dVcdt

def get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds,diffCoef=0.001,Cm=1.):
	'''returns dt, arr39, one_step_explicit_synchronous_splitting_w_Istim
	precomputes lookup table, arr39 and returns a jit compiling one_step method
	Example Usage:
	dt, kernelA, kernelB=get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds)
	'''
	#precompute lookup table
	arr39=get_arr39(dt,nb_dir)
	v_values=arr39[:,0]
	v0=np.min(v_values)
	dv=np.around(np.mean(np.diff(v_values)),1)
	lookup_params=_get_lookup_params(v_values,arr39)
	#     comp_dVcdt=get_comp_dVcdt(width=width, height=height, diffCoef=diffCoef, ds=ds, Cm=Cm)
	comp_ionic_flux=_get_comp_ionic_flux()
	#spatial discretization
	cddx = float(width / ds)  #if this is too big than the simulation will blow up (at a given timestep)
	cddy = float(height / ds) #if this is too big than the simulation will blow up (at a given timestep)
	cddx *= cddx
	cddy *= cddy
	_laplacian=get__laplacian(width,height,cddx,cddy)

	#     def one_step_explicit_synchronous_splitting_w_Istim_kernelA(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,txt_Istim):
	# @cuda.jit('void(float64[:,:,:], float64[:,:])', device=True, inline=True)
	@njit('void(float64[:,:,:], float64[:,:])')#,parallel=True)
	def one_step_explicit_synchronous_splitting_w_Istim_kernelA(txt,txt_Istim):
		'''advances voltage and intercellular calcium channels by dt/2 ms using forward euler integration
		for each pixel:
		advances V and Ca_i by dt/2 for each pixel using forward euler integration
		and then,
		for each pixel:
		advances gating variables using the exact flow map resulting from V
		advances V and Ca_i by dt/2 for each pixel using forward euler integration
		enforces agreement between inVc and outVc and between inmhjdfx and outmhjdfx (to avoid any confusion)
		'''
		#         inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
		#         inVc=txt[...,0:2]
		#         outVc=txt[...,2:4]
		#         inmhjdfx=txt[...,4:10]
		#         outmhjdfx=txt[...,10:16]
		#         dVcdt=txt[...,16:18]
		#
		d_arr_interp=np.zeros(shape=(width,height,21),dtype=float64)
		d_dVcdt = np.zeros(shape=(width, height, 2), dtype=float64)
		# d_arr_interp = cuda.shared.array((width,height,21),dtype=float64)
		# d_dVcdt = cuda.shared.array(shape=(width, height, 2), dtype=float64)

		# tx = cuda.threadIdx.x
		#
		# ty = cuda.threadIdx.y
		# bx = cuda.blockIdx.x
		# by = cuda.blockIdx.y
		# bw = cuda.blockDim.x
		# bh = cuda.blockDim.y
		#
		# x = tx + bx * bw
		# y = ty + by * bh
		# if x < width and y < height:
		for x in range(width):
			for y in range(height):
				#extract local variables
				Vc = txt[x,y,0:2]; V=Vc[0]
				inCgate = txt[x,y,4:10]

				#parse the row linearly interpolated from lookup table
				arr_interp=d_arr_interp[x,y]
				lookup_params(V,arr_interp)
				IK1T=float(arr_interp[13])    # 'xttab',
				x1=float(arr_interp[14])    # 'x1',

				#half step voltage and calcium
				#                 dVcdt_val=comp_dVcdt(inVc, x, y, inCgate, IK1T, x1)
				V=Vc[0];Ca_i = Vc[1]
				# Compute Ionic Current Density
				dVcdt=d_dVcdt[x,y]#np.array([0.,0.])
				comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i,dVcdt)
				Iion=dVcdt[0]
				dCa_i_dt=dVcdt[1]
				# Get the external stimulus current
				Istim = txt_Istim[x,y]
				# Compute transient term for transmembrane voltage
				dVltdt  = _laplacian(txt, x, y, V)
				# dVltdt = 0.
				# dVltdt_laplacian(txt, x, y, V, dVltdt)
				dVltdt *= float(diffCoef)
				dVltdt -= float((Iion+Istim)/Cm)
				dVcdt_val=dVcdt=txt[x,y,16:18]
				dVcdt_val[0]=dVltdt
				dVcdt_val[1]=dCa_i_dt
				# np.array([dVltdt,dCa_i_dt],dtype=np.float64)
				#record the result
				outVc_val=txt[x,y,2:4]
				outVc_val[0]=Vc[0] + 0.5 * dt * dVcdt_val[0]
				outVc_val[1]=Vc[1] + 0.5 * dt * dVcdt_val[1]
				txt[x,y,2]=outVc_val[0]
				txt[x,y,3]=outVc_val[1]


	# @njit# @cuda.jit('void(float64[:,:], float64[:,:])')
	# @njit
	# @cuda.jit('void(float64[:,:,:], float64[:,:])', device=True, inline=True)
	@njit('void(float64[:,:,:], float64[:,:])')#,parallel=True)
	def one_step_explicit_synchronous_splitting_w_Istim_kernelB(txt,txt_Istim):
		'''
		for each pixel:
		advances V and Ca_i by dt/2 for each pixel using forward euler integration
		and then,
		for each pixel:
		advances gating variables using the exact flow map resulting from V
		advances V and Ca_i by dt/2 for each pixel using forward euler integration
		enforces agreement between inVc and outVc and between inmhjdfx and outmhjdfx (to avoid any confusion)
		'''
		# width=txt.shape[0]
		# height=txt.shape[1]
		d_arr_interp=np.zeros(shape=(width,height,21),dtype=float64)
		d_dVcdt = np.zeros(shape=(width, height, 2), dtype=float64)
		for x in range(width):
			for y in range(height):
				#parse the row linearly interpolated from lookup table with updated voltage
				inCgate  = txt[x,y, 4:10]
				outCgate = txt[x,y,10:16]
				Vc = txt[x,y,2:4]; V  = Vc[0]
				arr_interp=d_arr_interp[x,y]
				lookup_params(V,arr_interp)
				# arr_interp=lookup_params(V)
				# x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
				IK1T=float(arr_interp[13])    # 'xttab',
				x1=float(arr_interp[14])    # 'x1',

				#full step the gating variables of step size dt (dt is encoded in arr39)
				comp_exact_next_gating_var(inCgate,outCgate,arr_interp)
				txt[x,y, 4:10]=outCgate
				txt[x,y,10:16]=outCgate
				# txt[x,y,4]=outCgate[0];txt[x,y,5]=outCgate[1];txt[x,y,6]=outCgate[2];
				# txt[x,y,7]=outCgate[3];txt[x,y,8]=outCgate[4];txt[x,y,9]=outCgate[5];
				# txt[x,y,10]=outCgate[0];txt[x,y,11]=outCgate[1];txt[x,y,12]=outCgate[2];
				# txt[x,y,13]=outCgate[3];txt[x,y,14]=outCgate[4];txt[x,y,15]=outCgate[5];

				#half step voltage and calcium
				# dVcdt_val=comp_dVcdt(inVc, x, y, inCgate, IK1T, x1)
				V=Vc[0];Ca_i = Vc[1]
				# Compute Ionic Current Density
				dVcdt=d_dVcdt[x,y]#np.array([0.,0.])
				comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i,dVcdt)
				Iion=dVcdt[0]
				dCa_i_dt=dVcdt[1]
				# Get the external stimulus current
				Istim = txt_Istim[x,y]
				# Compute transient term for transmembrane voltage
				dVltdt  = _laplacian(txt, x, y, V)
				# dVltdt = 0.
				# _laplacian(txt, x, y, V, dVltdt)
				dVltdt *= float(diffCoef)
				dVltdt -= float((Iion+Istim)/Cm)
				dVcdt_val=dVcdt=txt[x,y,16:18]
				dVcdt_val[0]=dVltdt
				dVcdt_val[1]=dCa_i_dt
				# dVcdt_val=np.array([dVltdt,dCa_i_dt])
				#record the result
				outVc_val=txt[x,y,2:4]
				outVc_val[0]=Vc[0] + 0.5 * dt * dVcdt_val[0]
				outVc_val[1]=Vc[1] + 0.5 * dt * dVcdt_val[1]
				txt[x,y,0]=outVc_val[0]
				txt[x,y,1]=outVc_val[1]
				txt[x,y,2]=outVc_val[0]
				txt[x,y,3]=outVc_val[1]

				#compute the current voltage/sodium flow map
				Vc = outVc_val; V  = Vc[0]
				arr_interp=d_arr_interp[x,y]
				lookup_params(V,arr_interp)
				# arr_interp=lookup_params(V)
				# x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
				IK1T=float(arr_interp[13])    # 'xttab',
				x1=float(arr_interp[14])    # 'x1',
				# dVcdt_val=comp_dVcdt(outVc, x, y, outCgate, IK1T, x1)
				#record rate of change of voltage and calcium current
				#                 dVcdt_val=comp_dVcdt(inVc, x, y, inCgate, IK1T, x1)
				V=Vc[0];Ca_i = Vc[1]
				# Compute Ionic Current Density
				dVcdt=d_dVcdt[x,y]
				comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i,dVcdt)
				Iion=dVcdt[0]
				dCa_i_dt=dVcdt[1]
				# Get the external stimulus current
				Istim = txt_Istim[x,y]
				# Compute transient term for transmembrane voltage
				dVltdt  = _laplacian(txt, x, y, V)
				# dVltdt = 0.
				# _laplacian(txt, x, y, cddx, cddy, V, dVltdt)
				dVltdt *= float(diffCoef)
				dVltdt -= float((Iion+Istim)/Cm)
				# dVcdt_val=np.array([dVltdt,dCa_i_dt])
				txt[x,y,16]=dVltdt
				txt[x,y,17]=dCa_i_dt

	kernelA=one_step_explicit_synchronous_splitting_w_Istim_kernelA
	kernelB=one_step_explicit_synchronous_splitting_w_Istim_kernelB
	return dt, kernelA, kernelB

def get_forward_integrate_kernel(nb_dir,dt,width,height,ds,diffCoef=0.001,Cm=1.):
	dt, kernelA, kernelB = get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds,diffCoef=diffCoef,Cm=Cm)
	# @cuda.jit('void(float64[:,:,:], float64[:,:], int32)')
	@njit('void(float64[:,:,:], float64[:,:], int32)')
	def forward_integrate_kernel(txt,txt_Istim, num_steps):
		for n in range(num_steps):
			kernelA(txt,txt_Istim)
			kernelB(txt,txt_Istim)

	return forward_integrate_kernel

def get_arr39(dt,nb_dir):
	cwd=os.getcwd()
	#generate lookup tables for timestep
	os.chdir(os.path.join(nb_dir,'lib/model'))
	cmd=f"python3 gener_table.py {dt}"
	os.system(cmd)
	#load lookup table for constant timestep, dt.
	os.chdir(os.path.join(nb_dir,'lib/model','lookup_tables'))
	# table_fn=f"luo_rudy_dt_{dt}.npz"
	# table_data=np.load(table_fn)
	# #convert table_data to a numpy array
	# kwds=table_data.get('kwds')
	# cols=kwds[-1].split('_')[1:]
	# keys=list(table_data.keys())
	# arr39=table_data[keys[-1]].T
	table_fn=f"luo_rudy_dt_{dt}_arr39.csv"
	arr39=pd.read_csv(table_fn,header=None).values
	#return to original working directory
	os.chdir(cwd)
	return arr39

# def get_arr39(dt,nb_dir):
# 	cwd=os.getcwd()
# 	dt=float(dt)
# 	#generate lookup tables for timestep
# 	os.chdir(os.path.join(nb_dir,'lib/model'))
# 	cmd=f"python3 gener_table.py {dt}"
# 	os.system(cmd)
# 	#load lookup table for constant timestep, dt.
# 	os.chdir(os.path.join(nb_dir,'lib/model','lookup_tables'))
# 	table_fn=f"luo_rudy_dt_{dt}.npz"
# 	table_data=np.load(table_fn)#,allow_pickle=True)
# 	#convert table_data to a numpy array
# 	kwds=table_data.get('kwds')
# 	cols=kwds[-1].split('_')[1:]
# 	keys=list(table_data.keys())
# 	arr39=table_data[keys[-1]].T
# 	#return to original working directory
# 	os.chdir(cwd)
# 	return arr39

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
	"""writes updated gating variables to outCgate,
	as determined by inCgate and parameterized by arr_interp."""
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


def _get_comp_ionic_flux(GNa=16.,GK1=0.6047,Gsi=0.052,EK1=-87.94,Eb=-59.87,ENa=54.4,GK=0.423):
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
	# @njit
	# @cuda.jit('void(float64, float64[:], float64, float64, float64, float64[:])', device=True, inline=True)
	@njit
	def comp_ionic_flux(V,inCgate,IK1T,x1,Ca_i,dVcdt):
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
		# if Ca_i<10**-6:
		# 	Ca_i=10**-6
		# Esi=7.7-13.0287*np.log(Ca_i)#mV  #from Luo1990.pdf
		Esi=-82.3-13.0287*np.log(Ca_i)#mV  #from lr_d0.f (WJ)
		Isi=Gsi*d*f*(V-Esi)
		#time dependent potassium current
		IK=x_var*x1#GK*x_var*x1#
		#total electric current
		Iion=INa+IK1T+Isi+IK
		#calcium uptake rate (dominated by activity of the sarcoplasmic reticulum)
		# dCa_i_dt=-10**-4*Isi+0.07*(10**-4-Ca_i) #mM #from Luo1990.pdf
		dCa_i_dt=-10**-7 * Isi + 0.07*(10**-7 - Ca_i)   #M  #from lr_d0.f (WJ)
		dVcdt[0]=float(Iion)
		dVcdt[1]=float(dCa_i_dt)#= np.array((Iion, dCa_i_dt))#,dtype=np.float64)#INa,IK1T,Isi,IK# GNa,m**3*h*j,(V-ENa)#,#
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

def _get_lookup_params(v_values,arr39):
	comp_row=get_comp_v_row(v_values)
	dv=np.around(np.mean(np.diff(v_values)),1)
	# d_arr39 = cuda.to_device(arr39, stream)
	chnlno=arr39.shape[-1]
	@njit
	def lookup_params(V,arr_interp):
		M=comp_row(V)
		arr=arr39[M:M+2]
		Vlo=arr[0,0]
		frac=(V-Vlo)/dv
		crac=1.-frac
		# arr_interp=frac*arr[1,:]+(1.-frac)*arr[0,:]
		# assert ( np.isclose( arr_interp[0]-V, 0.))#passed
		##alternative method for gpu
		a1= arr[1,:];a0= arr[0,:];
		for chnl in range(chnlno):
			arr_interp[chnl]=frac*a1[chnl]+crac*a0[chnl]
		# return arr_interp
	return lookup_params

# /*------------------------------------------------------------------------
#  * periodic boundary conditions for each read from textures
#  *------------------------------------------------------------------------
#  */
def get____pbc(width):
	# @cuda.jit('void(float64)', device=True, inline=True)
	@njit#('float64(float64)')
	def _pbc(x):
		'''writes to x,y to obey periodic boundary conditions
		(x, y) pixel coordinates of texture with values 0 to 1.
		tight boundary rounding is in use.'''
		if ( x < 0  ):				# // Left P.B.C.
			x = int(width - 1)
		elif ( x > (width - 1) ):	# // Right P.B.C.
			x = int(0)
		return x
	return _pbc

# @njit
# def pbc(S,x,y):
# 	'''S=texture with size 512,512,3
# 	(x, y) pixel coordinates of texture with values 0 to 1.
# 	tight boundary rounding is in use.'''
# 	width  = int(S.shape[0])
# 	height = int(S.shape[1])
# 	if ( x < 0  ):				# // Left P.B.C.
# 		x = width - 1
# 	elif ( x > (width - 1) ):	# // Right P.B.C.
# 		x = 0
# 	if( y < 0 ):				# //  Bottom P.B.C.
# 		y = height - 1
# 	elif ( y > (height - 1)):	# // Top P.B.C.
# 		y = 0
# 	return S[x,y]

# /*-------------------------------------------------------------------------
#  * Laplacian
#  *-------------------------------------------------------------------------
#  */
def get__laplacian(width,height,cddx,cddy):
	pbcx=get____pbc(width)
	pbcy=get____pbc(height)
	# @cuda.jit('float64(float64[:,:,:], int32, int32, float64)', device=True, inline=True)
	@njit
	def _laplacian(inVfs, x, y, V):
		rx=x+1; lx=x-1
		ry=y+1; ly=y-1
		rx=pbcx(rx);lx=pbcx(lx);
		ry=pbcy(ry);ly=pbcy(ly);
		#five point stencil
		dVltdt = float(
			(inVfs[ rx, y,0] - 2.0 * V +
			 inVfs[ lx, y,0]) * cddx +
			(inVfs[ x, ry,0] - 2.0 * V +
			 inVfs[ x, ly,0]) * cddy)
		return dVltdt
	return _laplacian
# @njit
# def laplacian(inVfs, x, y, cddx, cddy, V):
# 	#five point stencil
# 	dVltdt = (
# 	    (pbc(inVfs, x + 1, y)[0] - 2.0 * V +
# 	     pbc(inVfs, x - 1, y)[0]) * cddx +
# 	    (pbc(inVfs, x, y + 1)[0] - 2.0 * V +
# 	     pbc(inVfs, x, y - 1)[0]) * cddy)
# 	return dVltdt
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
#  * Caution: Adaptive time steping
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
	#as described previously [CITE one of those older papers]
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
