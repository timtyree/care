#!/bin/bash/env python3
#Timothy Tyree
#1.4.2021
# The model from Luo-Rudy (1990) modified as described 
# in Qu et alli (2000) to exhibit spiral defect chaos
# implemented on a square computational domain.
#TODO: use lookup table instead of using comp_ionic_currents
#TODO: ask WJ if he knows of any analytic solutions for the parameters of the FK model in terms of parameters of the LR model
from numba import njit, jit
import numpy as np

method='njit'
if method=='njit':
	njitsu = njit
if method=='cuda':
	import numba.cuda.njit as njitsu

# /*------------------------------------------------------------------------
#  * helper functions
#  *------------------------------------------------------------------------
#  */
# @njit
# def Tanh(x):
# 	'''fast/simple approximatation of the hyperbolic tangent function'''
# 	if ( x < -3.):
# 		return -1.
# 	elif ( x > 3. ):
# 		return 1.
# 	else:
# 		return x*(27.+x*x)/(27.+9.*x*x)
# # step function
# @njit
# def step(a,b):
# 	return 1 if a<=b else 0 # nan yields 1
# # return 0 if a>b else 1 # nan yields 0

# /*------------------------------------------------------------------------
#  * periodic boundary conditions for each read from textures
#  *------------------------------------------------------------------------
#  */
# @njitsu
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

# @njitsu
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


# /*------------------------------------------------------------------------
#  * comp transient gating variables
#  *------------------------------------------------------------------------
#  */
# @njitsu
def comp_transient_gating_variable(var, tau, varinfty):
	return (varinfty - var)/tau

# /*------------------------------------------------------------------------
#  * comp transient intercellular calcium concentration
#  *------------------------------------------------------------------------
#  */
# @njitsu
def comp_transient_intracellular_calcium_LR(Isi, Ca_i):
	#intracellular calcium concentration
	dCa_i_dt = -10**-4*Isi + 0.07 * (10**-4 - Ca_i)
	return dCa_i_dt

# /*------------------------------------------------------------------------
#  * comp transient voltage at pixel
#  *------------------------------------------------------------------------
#  */

def get_comp_rate_of_change_at_pixel_LR(ds = 0.015, width =200, height=200, 
	Cm=1., diffCoef=0.001,	Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,
	method='njit', **kwargs):
	"""Returns jit compiling function comp_rate_of_change_at_pixel_LR
	#default spatial discretization
	ds = 0.015 #cm/pixel
	width =200 #pixels
	height=200 #pixels

	#default electrical properties of myocardial tissue 
	Cm=1. # µF/cm^2 membrane capacitance determined by gap junctions between myocardial cells
	diffCoef=0.001 #cm^2/ms diffusion constant determined by membrane resistance
	

	#default ionic concentrations from
	Na_i = 18  #mM
	Na_o = 140 #mM
	K_i  = 145 #mM
	K_o  = 5.4 #mM
	Ca_o = 1.8 #mM
	EK1 = 10**3 * R*T/F * np.log (K_o/K_i) #mV

	# Ca_i varies during the action potential
	# Ca_i_initial = 2*10**-4 #mM

	Example Usage:
	comp_rate_of_change_at_pixel_LR=get_comp_rate_of_change_at_pixel_LR(ds = 0.015, width =200, height=200, 
	Cm=1., diffCoef=0.001,	Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,
	method='njit')"""

	# #other parameters
	# R = 8.3145  # J/(mol * °K) universal gas constant 
	# T = 273.15+37#°K physiologically normal body temperature 37°C
	# F = 96485.3321233100184 # C/mol faraday's constant
	# EK1 = 10**3 * R*T/F * np.log (K_o/K_i) #mV
	# # EK1 = -87.94 #mV
	# ENa = 

	#spatial discretization
	cddx = width  / ds  #if this is too big than the simulation will blow up (at a given timestep)
	cddy = height / ds #if this is too big than the simulation will blow up (at a given timestep)
	cddx *= cddx
	cddy *= cddy
	

	#physical parameters
	R = 8.3145  # J/(mol * °K) universal gas constant 
	T = 273.15+37#°K physiologically normal body temperature 37°C
	F = 96485.3321233100184 # C/mol faraday's constant
	

	EK1 = 10**3 * R*T/F * np.log (K_o/K_i) #mV
	comp_ionic_currents = get_comp_ionic_currents(K_i=K_i,K_o=K_o,ENa=54.4,EK=-77.0,method=method,**kwargs)
	comp_gating_constants = get_comp_gating_constants(method=method)

	if method=='njit':
		njitsu = njit
	if method=='cuda':
		import numba.cuda.njit as njitsu
	# @njitsu
	def comp_rate_of_change_at_pixel_LR(inVmhjdfxc, x, y):

		##################################################
		# Read Channels from Buffer
		##################################################
		C = pbc(inVmhjdfxc, x, y)
		V = C[0] #mV transmembrane voltage 
		#gating variables
		m = C[1] #activation gate parameter (Na)
		h = C[2] #fast inactivation gate parameter (INa)
		j = C[3] #slow inactivation gate parameter (INa)
		d = C[4] #activation gate parameter (Isi)
		f = C[5] #inactivation gate parameter (Isi)
		x_var = C[6] #activation gate parameter (IK)
		Ca_i = C[7] # calcium concentration inside the cell 
		# Ca_i = pbc(inCa_i, x, y)[0]
		
		##################################################
		# Compute Ionic Current Density
		##################################################
		Iion, dCa_i_dt = comp_ionic_currents(V, m, h, j, d, f, x_var, Ca_i)

		##################################################
		# Compute transient term for transmembrane voltage
		##################################################
		dVltdt=laplacian(inVmhjdfxc, x, y, cddx, cddy, V)	
		dVltdt *= float(diffCoef)
		dVltdt -= float(Iion/Cm)

		retval=comp_gating_constants(V)
		tau_m, tau_h, tau_j, tau_d, tau_f, tau_x, m_infty, h_infty, j_infty, d_infty, f_infty, x_infty=retval

		##################################################
		# Compute transient term for gating variables
		##################################################
		dmdt = comp_transient_gating_variable(m, tau_m, m_infty)
		dhdt = comp_transient_gating_variable(h, tau_h, h_infty)
		djdt = comp_transient_gating_variable(j, tau_j, j_infty)
		dddt = comp_transient_gating_variable(d, tau_d, d_infty)
		dfdt = comp_transient_gating_variable(f, tau_f, f_infty)
		dxdt = comp_transient_gating_variable(x_var, tau_x, x_infty)

		##################################################
		# Return rate of change of channels
		##################################################
		txt_out = np.array([
			dVltdt,
			dmdt,
			dhdt,
			djdt,
			dddt,
			dfdt,
			dxdt,
			dCa_i_dt
			],dtype=np.float64)
		return txt_out
	return comp_rate_of_change_at_pixel_LR



# We assume that a short term stimulation does not 
# appreciably affect the ionic environment of the 
# cell under normal conditions and, therefore, the 
# ionic concentrations (except Ca_i) do not change 
# dynamically in our simulations.


# #Qu changes the other maximum conductances and relaxation time constants to create different spiral wave behaviors
# # Unless explicitly stated either in the text or in the figure captions, parameter values are the same as specified in the original LR1 model.

# #temporal discretization
# # in Qu (2000), the 2D simulation was conducted using operator splitting and adaptive time step methods
# # The ordinary differential equations were in-
# # tegrated with a time step which varied from 0.005 to 0.1 ms
# dt_min = 0.005 #ms
# dt_max = 0.1   #ms

# # In Qu2000.pdf, the partial differential equation was integrated
# # using the alternating direction implicit method with a time step of 0.1 ms.

# @njitsu
def comp_next_time_step(dt_prev, dV):
	''' returns the size of the next time step
	Adaptive time stepping as described in Luo1990.pdf.
	During the stimulus, a fixed time step (0.05 or 0.01 msec) 
	should be used to minimize variability in the stimulus duration 
	caused by the time discretizationprocedure.
	'''
	dVmax = 0.8 #mV
	dVmin = 0.2 #mV
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


def get_comp_ionic_currents(K_i=145.0,K_o=5.4,ENa=54.4,EK=-77.0,method='njit',**kwargs):
	'''
	from 1990Luo.pdf modified as described in Qu2000.pdf to support spiral wave breakup.
	let potassium levels take default constant values.
	K_i  = 145 #mM
	K_o  = 5.4 #mM
	'''
	#maximum conductances
	GNa = 16.     #mS/cm^2 from Qu2000.pdf #GNa=23 in Luo1990.pdf
	GK1 = 0.6047  #mS/cm^2 from Qu2000.pdf
	Gsi = 0.052   #mS/cm^2 spiral wave breakup phase from Qu2000.pdf
	GK  = 0.423   #mS/cm^2 #from Qu2000.pdf
	#Gsi = 0.038  #mS/cm^2  chaotic meander phase from Qu2000.pdf
	#Gsi = 0.     #mS/cm^2 quasiperiodic phase from Qu2000.pdf
	# Note the strong inward rectification, the 
	# crossover between curves of different [K]o, the
	# zero contribution at high potentials, and the 
	# negative slope over a certain potential range.
	# # GKp = 0.0183  #mS/cm^2
	# # Gb = 0.03921  #mS/cm^2 background conductance
	# # GK1 = 0.6047 * np.sqrt (K_o/5.4)
	# GK = 0.282 * np.sqrt(K_o/5.4) from Luo1990.pdf.  This value is obtained from the fuly activated current IK=IK/X(B-Rmodel,3Figure1)forV= -100 mV, a potential at which Xi= 1.
	# Gsi = 0.09
	#other parameters
	R = 8.3145  # J/(mol * °K) universal gas constant 
	T = 273.15+37#°K physiologically normal body temperature 37°C
	F = 96485.3321233100184 # C/mol faraday's constant

	#reversal potentials
	#reversal potentials are computed from Nernst potential 
	# ENa = 54.4 #mV
	# PRNaK=0.01833 #the permeability ratio of sodium to potassium (see Luo1990.pdf)
	# ENa = RT/F * np.log(Na_o/Na_i)
	# EK = RT/F * np.log((K_o + PRNaK * Na_o)/(K_i + PRNaK * Na_i))
	# EK  = -77. #mV #from Qu2000.pdf
	EK1 = R*T/F * np.log (K_o/K_i)
	EKp = EK1
	# EK1 = -87.94
	Eb = -59.87 #mV


	if method=='njit':
		njitsu = njit
	if method=='cuda':
		import numba.cuda.njit as njitsu

	#precompute parameters
	# @njitsu
	def comp_ionic_currents(V, m, h, j, d, f, x, Ca_i):
		'''Returns the electrical transmembrane current flux 
		in units of µA/cm^2 using the model parameters described
		in Table 1 of Luo1990.pdf, wherein
		the ionic currents were computed according to

		#fast inward sodium current
		INa = GNa * m**3 * h * j * (V - ENa)
		#slow inward current, assumed to be the L-type calcium current
		Isi = Gsi * d * f * (V - Esi)
		#slow outward time-dependent potassium􏰡 current;
		IK  = GK  * x * x1 * (V - EK)
		#time dependent potassium current
		IK1 = GK1 * K1infty * (V - EK1)
		#plateau potassium current
		IKp = GKp * Kp * (V - EKp)
		#total background current
		Ib  = Gb * (V  - Eb)
		#"total time independent potassium current"
		# IK1T = IK1 + IKp + Ib
		#total ionic current density 
		Iion = INa + Isi + IK + IK1 + IKp + Ib
		'''
		####################
		# Inward Currents
		####################
		#Fast sodium current
		INa = GNa*m**3*h*j*(V-ENa)

		#Slow inward current
		Esi=7.7-13.0287*np.log(Ca_i)#mV
		Isi=Gsi*d*f*(V-Esi)
		#calcium uptake rate (dominated by activity of the sarcoplasmic reticulum)
		dCa_i_dt=-10**-4*Isi+0.07*(10**-4-Ca_i)
		# dCa_i_dt=comp_transient_intracellular_calcium_LR(Isi, Ca_i):

		####################
		# Outward Currents
		####################
		#Time-dependent potassium current
		if V>-100:#mV
			x1=2.837*(np.exp(0.04*(V+77.0))-1.0)/(V+77.0)*np.exp(0.04*(V+35.0))
		else:
			x1=1.0
		IK=GK*x*x1*(V-EK)

		#Time-independent potassium parameters
		aK1=1.02/(1.0+np.exp(0.2385*(V-EK1-59.215)))
		bK1=(0.49124*np.exp(0.08032*(V-EK1-5.476)))+np.exp(0.06175*(V-EK1-594.31))/(1.0+np.exp(-0.5143*(V-EK1+4.753)))
		K1infty = aK1/(aK1 + bK1)
		#fast potassium current
		IK1=GK1*K1infty*(V-EK1)

		#Plateau potassium current
		Kp=1.0/(1.0+np.exp((7.488-V)/5.98))
		IKp=0.0183*Kp*(V-EKp)
		#Background Current
		Ib=0.03921*(V+59.87)
		#Total time-independent potassium current
		IK1T=IK1+IKp+Ib

		####################
		# Return net ionic current
		####################
		Iion=INa+IK1T+Isi
		return Iion, dCa_i_dt
	return comp_ionic_currents


def get_comp_gating_constants(method='njit'):#**kwargs):
	#precompute parameters
	if method=='njit':
		njitsu = njit
	if method=='cuda':
		import numba.cuda.njit as njitsu

	# @njitsu
	def comp_gating_constants(V):
		'''from Table 1 of Luo1990.pdf
		ay is the rate constant for y ion channels/gates opening (msec).
		by is the rate constant for y ion channels/gates closing (msec).
		'''
		####################
		# Inward Ion Gates
		####################
		#Fast sodium current
		am=0.32*(V+47.13)/(1.0-np.exp(-0.1*(V+47.13)))
		bm=0.08*np.exp(-V/11.)
		if V>=-40.0:#mV sodium gates are closed
			ah=0.0 
			aj=0.0
			bh=1.0/(0.13*np.exp((V+10.66)/-11.1))
			bj=0.3*np.exp(-2.535*10**-7*V)/(1.0+np.exp(-0.1*(V+32.0)))
		else:
			ah=0.135*np.exp((80.0+V)/-6.8)
			aj=(-1.2714*10**5*np.exp(0.2444*V)-3.474*10**-5*np.exp(-0.04391*V))*(V+37.78)/(1.0+np.exp(0.311*(V+79.23)))
			bh=3.56*np.exp(0.079*V)+3.1*10**5*np.exp(0.35*V)
			bj=0.1212*np.exp(-0.01052*V)/(1.0+np.exp(-0.1378*(V+40.14)))

		#Slow inward current
		ad=0.095*np.exp(-0.01*(V-5.0))/(1.0+np.exp(-0.072*(V-5.0)))
		bd=0.07*np.exp(-0.017*(V+44.0))/(1.0+np.exp(0.05*(V+44.0)))
		af=0.012*np.exp(-0.008*(V+28.0))/(1.0+np.exp(0.15*(V+28.0)))
		bf=0.0065*np.exp(-0.02*(V+30.0))/(1.0+np.exp(-0.2*(V+30.0)))

		####################
		# Outward Ion Gates
		####################
		#Time-dependent potassium current
		ax=0.0005*np.exp(0.083*(V+50.0))/(1.0+np.exp(0.057*(V+50.0)))
		bx=0.0013*np.exp(-0.06*(V+20.0))/(1.0+np.exp(-0.04*(V+20.0)))
		
		####################
		# Compute rates
		####################
		tau_m=1.0/(am+bm)
		tau_h=1.0/(ah+bh)
		tau_j=1.0/(aj+bj)
		tau_d=1.0/(ad+bd)
		tau_f=1.0/(af+bf)
		tau_x=1.0/(ax+bx)

		####################
		# Compute equilibria
		####################
		m_infty=am*tau_m
		h_infty=ah*tau_h
		j_infty=aj*tau_j
		d_infty=ad*tau_d
		f_infty=af*tau_f
		x_infty=ax*tau_x

		retval=np.array([tau_m, tau_h, tau_j, tau_d, tau_f, tau_x, 
			m_infty, h_infty, j_infty, d_infty, f_infty, x_infty
			])
		return retval
	return comp_gating_constants
