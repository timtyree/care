#!/bin/bash/env python3
from numba import njit
import numpy as np


#TODO (later): save canvas as a .txt file, keep it simple with numpy
# #(don't)TODO: find old output method to get 512,512 images from Dicty. Dispersal folder
# #(don't)TODO: test that the saved buffer didn't rescale the numerical values to 256.
# plt.figure(figsize=(4,4))
# plt.imshow(gimage[...,0].astype('uint8'))
# plt.axis('off')
# plt.layout_tight()
# plt.savefig('Figures/init.jpg', dpi=128)

# Image.frombuffer("L", (512, 512), gimage, 'raw', "L", 0, 1)     
@njit
def set_voltage_in_box(image, min_x, max_x, min_y, max_y, width, height, value=30.0):
	for x in range(width):
		for y in range(height):
			if min_x <= x < max_x and min_y <= y < max_y:
				image[y, x, 0] = value

@njit
def init_in_box(image, min_x, max_x, min_y, max_y, width, height, value=30.0):
	for x in range(width):
		for y in range(height):
			if min_x <= x < max_x and min_y <= y < max_y:
				image[y, x, 0] = value
				image[y, x, 1] = 0.#0.#11116473
				image[y, x, 2] = 0.#0.#02320262
			else:
				image[y, x, 0] = 0.0#01574451
				image[y, x, 1] = 1.0#11116473
				image[y, x, 2] = 0.4#02320262

def initialize_mesh(width,height,channel_no, value, zero=None):
	'''create initialization buffer for the standard.  
	let the ring propagate out until tissue in the center 
	is excitable before exploring initial trajectories based 
	on the width of rectangular perturbations.'''
	if zero is None:
		zero = np.zeros((width, height, channel_no), dtype = np.float64)
	gimage = zero.copy()
	# change a rectangle to initial values
	init_in_box(gimage, 
					 min_x=256-64, 
					 max_x=256+64, 
					 min_y=256-32, 
					 max_y=256+32, 
					 width=width, 
					 height=height, 
					 value=value
					)    
	return gimage          

# @njit
# def set_to_value_in_box(image, min_x, max_x, min_y, max_y, width, height, value=30.0):
# 	for x in range(width):
# 		for y in range(height):
# 			if min_x <= x < max_x and min_y <= y < max_y:
# 				image[y, x, 0] = value
# 				image[y, x, 1] = 0.#0.#11116473
# 				image[y, x, 2] = 0.#0.#02320262
# 			else:
# 				image[y, x, 0] = 0.0#01574451
# 				image[y, x, 1] = 1.0#11116473
# 				image[y, x, 2] = 0.4#02320262

# def initialize_mesh(width,height,channel_no, value):
# 	#create standardized initialization buffer
# 	gimage = np.zeros((width, height, channel_no), dtype = np.float64)
# 	# change a rectangle to initial values
# 	set_to_value_in_box(gimage, 
# 					 min_x=256-64, 
# 					 max_x=256+64, 
# 					 min_y=256-32, 
# 					 max_y=256+32, 
# 					 width=width, 
# 					 height=height, 
# 					 value=value
# 					)    
# 	return gimage          

@njit
def Tanh(x):
	'''fast/simple approximatation of the hyperbolic tangent function'''
	if ( x < -3.):
		return -1.
	elif ( x > 3. ):
		return 1.
	else: 
		return x*(27.+x*x)/(27.+9.*x*x)

# /*------------------------------------------------------------------------
#  * applying periodic boundary conditions for each texture call
#  *------------------------------------------------------------------------
#  */ 

#S=texture with size 512,512,3
#(x, y) pixel coordinates of texture with values 0 to 1
@njit
def pbc(S,x,y):
	width  = 512
	height = 512
	if ( x < 1  ):#{ // Left P.B.C.
		x = width - 1  
	elif ( x > (width - 2) ):#{ // Right P.B.C.
		x = 0 
	if( y <  1 ):#{}    //  Bottom P.B.C.
		y = height - 1
	elif ( y > (height - 2)):#{ // Top P.B.C.
		y = 0
	return S[x,y,:]

# step function
@njit
def step(a,b):
	return 1 if a<b else 0 # nan yields 1
# return 0 if a>b else 1 # nan yields 0

# /*------------------------------------------------------------------------
#  * time step at a pixel
#  *------------------------------------------------------------------------
#  */ 
@njit
def time_step_at_pixel(inVfs, x, y):#, h):
	# define parameters
	width  = 512
	height = 512
	ds_x   = 18 #domain size
	ds_y   = 18

	# dt = 0.1
	diffCoef = 0.001
	C_m = 1.0

	tau_pv = 3.33
	tau_v1 = 19.6
	tau_v2 = 1000
	tau_pw = 667
	tau_mw = 11
	tau_d  = 0.42
	tau_0  = 8.3
	tau_r  = 50
	tau_si = 45
	K      = 10
	V_sic  = 0.85
	V_c    = 0.13
	V_v    = 0.055
	C_si   = 1.0
	Uth    = 0.9

	# tau_pv = 3.33
	# tau_v1 = 15.6
	# tau_v2 = 5
	# tau_pw = 350
	# tau_mw = 80
	# tau_d = 0.407
	# tau_0 = 9
	# tau_r = 34
	# tau_si = 26.5
	# K = 15
	# V_sic = 0.45
	# V_c = 0.15
	# V_v = 0.04
	# C_si = 1
	# Uth = 0.9

	# dx, dy = (1, 1)
	#(1/512, 1/512)


	# cddx = 1 / ds_x  #no motion
	# cddy = 1 / ds_y  #no motion
	cddx = width / ds_x  #blows up  #TODO: retry with fast/slow initialized at unity
	cddy = height / ds_y #blows up

	cddx *= cddx
	cddy *= cddy

	# /*------------------------------------------------------------------------
	#  * reading from textures
	#  *------------------------------------------------------------------------
	#  */
	C = pbc(inVfs, x, y)
	vlt = C[0]
	#volts
	fig = C[1]
	#fast var
	sig = C[2]
	#slow var

	# /*-------------------------------------------------------------------------
	#  * Calculating right hand side vars
	#  *-------------------------------------------------------------------------
	#  */
	p = step(V_c, vlt)
	q = step(V_v, vlt)

	tau_mv = (1.0 - q) * tau_v1 + q * tau_v2

	Ifi = -fig * p * (vlt - V_c) * (1.0 - vlt) / tau_d
	Iso = vlt * (1.0 - p) / tau_0 + p / tau_r

	tn = Tanh(K * (vlt - V_sic))
	Isi = -sig * (1.0 + tn) / (2.0 * tau_si)
	Isi *= C_si
	dFig2dt = (1.0 - p) * (1.0 - fig) / tau_mv - p * fig / tau_pv
	dSig2dt = (1.0 - p) * (1.0 - sig) / tau_mw - p * sig / tau_pw

	#fig += dFig2dt * h
	#sig += dSig2dt * h

	# /*-------------------------------------------------------------------------
	#  * Laplacian
	#  *-------------------------------------------------------------------------
	#  */
	#     ii = np.array([1,0])  ;
	#     jj = np.array([0,1])  ;
	#     gamma = 1./3. ;
	dVlt2dt = (1. - 1. / 3.) * (
		(pbc(inVfs, x + 1, y)[0] - 2.0 * C[0] +
		 pbc(inVfs, x - 1, y)[0]) * cddx +
		(pbc(inVfs, x, y + 1)[0] - 2.0 * C[0] +
		 pbc(inVfs, x, y - 1)[0]) * cddy) + (1. / 3.) * 0.5 * (
			 pbc(inVfs, x + 1, y + 1)[0] + pbc(
				 inVfs, x + 1, y - 1)[0] + pbc(inVfs, x - 1, y - 1)[0] +
			 pbc(inVfs, x - 1, y + 1)[0] - 4.0 * C[0]) * (cddx + cddy)
	dVlt2dt *= diffCoef

	# /*------------------------------------------------------------------------
	#  * I_sum
	#  *------------------------------------------------------------------------
	#  */
	I_sum = Isi + Ifi + Iso

	# /*------------------------------------------------------------------------
	#  * Time integration for membrane potential
	#  *------------------------------------------------------------------------
	#  */

	dVlt2dt -= I_sum / C_m
	# vlt += dVlt2dt * dt

	# /*------------------------------------------------------------------------
	#  * ouputing the shader
	#  *------------------------------------------------------------------------
	#  */
	#     state  = (vlt,Ifi, Iso, Isi);
	# outVfs = (vlt, fig, sig)
	# return np.array((vlt, fig, sig),dtype=np.float64)
	return np.array((dVlt2dt,dFig2dt,dSig2dt),dtype=np.float64)

#     '''assuming width and height have the size of the first two axes of texture'''
@njit
def get_time_step(texture, out):
	#width  = 512
	#height = 512
	for x in range(512):
		for y in range(512):
			out[x,y] = time_step_at_pixel(texture,x,y)





