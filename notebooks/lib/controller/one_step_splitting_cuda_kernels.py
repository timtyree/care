#!/bin/bash/env python3
#Timothy Tyree
#1.16.2021
from numba import njit, cuda, float64
import numpy as np
import os
from .. import *
from math import log
# @njit
# @njit
def get____pbc(width):
	@cuda.jit('void(float64)', device=True, inline=True)
	def ___pbc(x):
		'''writes to x,y to obey periodic boundary conditions
		(x, y) pixel coordinates of texture with values 0 to 1.
		tight boundary rounding is in use.'''
		if ( x < 0  ):				# // Left P.B.C.
			x = width - 1
		elif ( x > (width - 1) ):	# // Right P.B.C.
			x = 0
		# if( y < 0 ):				# //  Bottom P.B.C.
		# 	y = height - 1
		# elif ( y > (height - 1)):	# // Top P.B.C.
		# 	y = 0
	return ___pbc

def get__laplacian(width,height,cddx,cddy):
	pbcx=get____pbc(width)
	pbcy=get____pbc(height)
	# @njit
	@cuda.jit('float64(float64[:,:,:], int32, int32, float64)', device=True, inline=True)
	def _laplacian(inVfs, x, y, V):
		rx=x+1; lx=x-1
		ry=y+1; ly=y-1
		pbcx(rx);pbcx(lx);
		pbcy(ry);pbcy(ly);
		#five point stencil
		dVltdt = float(
			(inVfs[ rx, y,0] - 2.0 * V +
			 inVfs[ lx, y,0]) * cddx +
			(inVfs[ x, ry,0] - 2.0 * V +
			 inVfs[ x, ly,0]) * cddy)
		return dVltdt
	return _laplacian
#(deprecated) nine point stencil
# 	dVlt2dt = (1. - 1. / 3.) * (
# 		(pbc(inVfs, x + 1, y)[0] - 2.0 * C[0] +
# 		 pbc(inVfs, x - 1, y)[0]) * cddx +
# 		(pbc(inVfs, x, y + 1)[0] - 2.0 * C[0] +
# 		 pbc(inVfs, x, y - 1)[0]) * cddy) + (1. / 3.) * 0.5 * (
# 			 pbc(inVfs, x + 1, y + 1)[0] + pbc(
# 				 inVfs, x + 1, y - 1)[0] + pbc(inVfs, x - 1, y - 1)[0] +
# 			 pbc(inVfs, x - 1, y + 1)[0] - 4.0 * C[0]) * (cddx + cddy)


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
	@cuda.jit('void(float64, float64[:], float64, float64, float64, float64[:])', device=True, inline=True)
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
		if Ca_i<10**-6:
			Ca_i=10**-6
		Esi=7.7-13.0287*log(Ca_i)#mV
		Isi=Gsi*d*f*(V-Esi)
		#time dependent potassium current
		IK=x_var*x1#GK*x_var*x1#
		#total electric current
		Iion=INa+IK1T+Isi+IK
		#calcium uptake rate (dominated by activity of the sarcoplasmic reticulum)
		dCa_i_dt=-10**-4*Isi+0.07*(10**-4-Ca_i)
		dVcdt[0]=float(Iion)
		dVcdt[1]=float(dCa_i_dt)#= np.array((Iion, dCa_i_dt))#,dtype=np.float64)#INa,IK1T,Isi,IK# GNa,m**3*h*j,(V-ENa)#,#
	return comp_ionic_flux

# def _get_comp_v_row(v_values):
# 	dv=np.around(np.mean(np.diff(v_values)),1)
# 	v0=np.min(v_values)
# 	# @njit
# 	@cuda.jit('void(float64,int32)')
# 	def comp_v_row(v, M):
# 		M = int((v-v0)/dv)
# 	return comp_v_row

def _get_lookup_params(v_values,arr39, stream):
	ndimv=v_values.shape[0]
	# dv=np.around(np.mean(np.diff(v_values)),1)
	# comp_row=_get_comp_v_row(v_values)
	dv=np.around(np.mean(np.diff(v_values)),1)
	v0=np.min(v_values)
	arr39shape=arr39.shape
	chnlno=arr39shape[-1]
	d_arr39 = cuda.to_device(arr39, stream)
	# d_arr39 = cuda.device_array_like(arr39, stream)
	# d_arr39 = cuda.const.array_like(arr39)
#     d_zero = np.zeros(chnlno,dtype=np.float64)
#     d_zero = cuda.to_device(np.zeros(chnlno,dtype='float64'))
#     @njit('void(float64,float64[:])')
	@cuda.jit('void(float64,float64[:])', device=True, inline=True)
	def lookup_params(V,arr_interp):
		M = int((V-v0)/dv) #comp_row
		if M>ndimv-1:
			M=ndimv-1
# 		arr=d_arr39[M:M+2]
# #         arr=arr39[M:M+2]
# 		Vlo=arr[0,0]
# 		a1= arr[1,:];a0= arr[0,:];

		# a1=d_arr39[M+1,:]
		# a0=d_arr39[M,:]
		# Vlo=d_arr39[M,0]

		#linear interpolation
		frac=float((V-d_arr39[M,0])/dv)
		crac=1.-frac
		for chnl in range(chnlno):
			arr_interp[chnl]=frac*d_arr39[M+1,chnl]+crac*d_arr39[M,chnl]
	return lookup_params

		# a1=d_arr39[M+1,:]
		# a0=d_arr39[M,:]
		# Vlo=d_arr39[M,0]
		#
		# #linear interpolation
		# frac=(V-Vlo)/dv
		# crac=1.-frac
		# for chnl in range(chnlno):
		# 	arr_interp[chnl]=frac*a1[chnl]+crac*a0[chnl]


#TODO: optimize by making a cuda.jit compiled _... for each function called in the kernels
def get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds,stream,diffCoef=0.001,Cm=1.):
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
	lookup_params=_get_lookup_params(v_values,arr39,stream)
	#     comp_dVcdt=get_comp_dVcdt(width=width, height=height, diffCoef=diffCoef, ds=ds, Cm=Cm)
	comp_ionic_flux=_get_comp_ionic_flux()
	#spatial discretization
	cddx = float(width / ds)  #if this is too big than the simulation will blow up (at a given timestep)
	cddy = float(height / ds) #if this is too big than the simulation will blow up (at a given timestep)
	cddx *= cddx
	cddy *= cddy
	_laplacian=get__laplacian(width,height,cddx,cddy)

	# zero_interp=cuda.to_device(np.zeros(21,dtype='float64'))
	# zero_2=cuda.to_device(np.zeros(2,dtype='float64'))

	# d_dVcdt=cuda.to_device(np.zeros((width,height,2),dtype='float64'))
	# d_arr_interp=cuda.to_device(np.zeros((width,height,21),dtype='float64'))



	#     def one_step_explicit_synchronous_splitting_w_Istim_kernelA(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,txt_Istim):
	# @njit# @cuda.jit('void(float64[:,:], float64[:,:])')
	# @njit
	@cuda.jit('void(float64[:,:,:], float64[:,:])', device=True, inline=True)
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
		d_arr_interp = cuda.shared.array((width,height,21),dtype=float64)
		d_dVcdt = cuda.shared.array(shape=(width, height, 2), dtype=float64)

		tx = cuda.threadIdx.x

		ty = cuda.threadIdx.y
		bx = cuda.blockIdx.x
		by = cuda.blockIdx.y
		bw = cuda.blockDim.x
		bh = cuda.blockDim.y

		x = tx + bx * bw
		y = ty + by * bh
		# for x in range(width):
		# 	for y in range(height):
		if x < width and y < height:
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

		# cuda.syncthreads()

	# @njit# @cuda.jit('void(float64[:,:], float64[:,:])')
	# @njit
	@cuda.jit('void(float64[:,:,:], float64[:,:])', device=True, inline=True)
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
		d_arr_interp = cuda.shared.array((width,height,21),dtype=float64)
		d_dVcdt = cuda.shared.array(shape=(width, height, 2), dtype=float64)

		tx = cuda.threadIdx.x

		ty = cuda.threadIdx.y

		bx = cuda.blockIdx.x
		by = cuda.blockIdx.y
		bw = cuda.blockDim.x
		bh = cuda.blockDim.y
		x = tx + bx * bw
		y = ty + by * bh

		# for x in range(width):
		# 	for y in range(height):
		if x < width and y < height:
			#parse the row linearly interpolated from lookup table with updated voltage
			inCgate  = txt[x,y,4:10]
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
			txt[x,y,4]=outCgate[0];txt[x,y,5]=outCgate[1];txt[x,y,6]=outCgate[2];
			txt[x,y,7]=outCgate[3];txt[x,y,8]=outCgate[4];txt[x,y,9]=outCgate[5];
			txt[x,y,10]=outCgate[0];txt[x,y,11]=outCgate[1];txt[x,y,12]=outCgate[2];
			txt[x,y,13]=outCgate[3];txt[x,y,14]=outCgate[4];txt[x,y,15]=outCgate[5];
			# txt[x,y,4:10]=outCgate
			# txt[x,y,10:16]=outCgate

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

		# cuda.syncthreads()
				#save texture to output
				# txt_nxt[x,y]=stack_pxl(outVc_val,outVc_val,outCgate,outCgate,dVcdt_val)

		# #copy out to in
		# inmhjdfx=outmhjdfx.copy()
		# outVc=inVc.copy()
		# np.stack(*(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)).T
	kernelA=one_step_explicit_synchronous_splitting_w_Istim_kernelA
	kernelB=one_step_explicit_synchronous_splitting_w_Istim_kernelB
	return dt, kernelA, kernelB

def get_forward_integrate_kernel(nb_dir,dt,width,height,ds,stream,diffCoef=0.001,Cm=1.):
	"""forward_integrate_kernel may raise numba jit compilation errors...
	Example Usage:
forward_integrate_kernel=get_forward_integrate_kernel(nb_dir,dt,width,height,ds,stream,diffCoef=0.001,Cm=1.)
	"""
	dt, kernelA, kernelB = get_one_step_explicit_synchronous_splitting_w_Istim_kernel(nb_dir,dt,width,height,ds,stream,diffCoef=diffCoef,Cm=Cm)

	@cuda.jit('void(float64[:,:,:], float64[:,:], int32)')
	def forward_integrate_kernel(txt,txt_Istim, num_steps):
		for n in range(num_steps):
			kernelA(txt,txt_Istim)
			cuda.syncthreads()
			kernelB(txt,txt_Istim)
			cuda.syncthreads()

	return forward_integrate_kernel
