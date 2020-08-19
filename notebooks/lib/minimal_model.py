#!/bin/bash/env python3
from numba import njit, jit
import numpy as np

# @njit
# def Tanh(x):
# 	'''fast/simple approximatation of the hyperbolic tangent function'''
# 	if ( x < -3.):
# 		return -1.
# 	elif ( x > 3. ):
# 		return 1.
# 	else:
# 		return x*(27.+x*x)/(27.+9.*x*x)

@njit
def Tanh(x):
    return np.math.tanh(x)
    
# /*------------------------------------------------------------------------
#  * applying periodic boundary conditions for each texture call
#  *------------------------------------------------------------------------
#  */
@njit
def pbc(S,x,y):
    '''S=texture with size width,height,3
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

@njit
def pbc1(S,x,y):
	'''S=texture with size width,height,1
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

# step function
@njit
def step(a,b):
    return 1 if a<=b else 0 # nan yields 1

# /*------------------------------------------------------------------------
#  * time step at a pixel
#  *------------------------------------------------------------------------
#  */
@njit
def time_step_at_pixel(inVfs, x, y):#, h):
    # define parameters
    width    = int(inVfs.shape[0])
    height   = int(inVfs.shape[1])
    DX       = 0.025 #cm/pxl
    DY       = 0.025 #cm/pxl
    cddx     = 1/DX**2
    cddy     = 1/DY**2
    diffCoef = 0.0005 # cm^2 / ms
    C_m      = 1.000  # 􏰎microFarad/cm^2 

    #parameter set 8 of FK model from Fenton & Cherry (2002)
    tau_pv   = 13.03
    tau_v1   = 19.6
    tau_v2   = 1250
    tau_pw   = 800
    tau_mw   = 40
    tau_d    = 0.45# also interesting to try, but not F&C8's 0.45: 0.407#0.40#0.6#
    tau_0    = 12.5
    tau_r    = 33.25
    tau_si   = 29#
    K        = 10
    V_sic    = 0.85#
    V_c      = 0.13
    V_v      = 0.04
    C_si     = 1  # I didn't find this (trivial) multiplicative constant in Fenton & Cherry (2002).  The value C_si = 1 was used in Kaboudian (2019).
    dx, dy   = (1, 1)# size of a pixel

    # /*------------------------------------------------------------------------
    #  * reading from textures
    #  *------------------------------------------------------------------------
    #  */
    C = pbc(inVfs, x, y)
    vlt = C[0]#volts
    fig = C[1]#fast var
    sig = C[2]#slow var

    # /*-------------------------------------------------------------------------
    #  * Calculating right hand side vars
    #  *-------------------------------------------------------------------------
    #  */
    p = step(V_c, vlt)
    q = step(V_v, vlt)
    
    #original FK model
    #     tau_mv = (1.0 - q) * tau_v1 + q * tau_v2
    #FK model with tau_v2 and tau_v1 switched
    tau_mv = (1.0 - q) * tau_v2 + q * tau_v1
    
    Ifi = -fig * p * (vlt - V_c) * (1.0 - vlt) / tau_d
    Iso = vlt * (1.0 - p) / tau_0 + p / tau_r

    tn = Tanh(K * (vlt - V_sic))
    Isi = -sig * (1.0 + tn) / (2.0 * tau_si)
    Isi *= C_si
    dFig2dt = (1.0 - p) * (1.0 - fig) / tau_mv - p * fig / tau_pv
    dSig2dt = (1.0 - p) * (1.0 - sig) / tau_mw - p * sig / tau_pw

    #five point stencil
    dVlt2dt = (
        (pbc(inVfs, x + 1, y)[0] - 2.0 * C[0] +
         pbc(inVfs, x - 1, y)[0]) * cddx +
        (pbc(inVfs, x, y + 1)[0] - 2.0 * C[0] +
         pbc(inVfs, x, y - 1)[0]) * cddy)
    dVlt2dt *= diffCoef
    
    #(deprecated) nine point stencil
    # 	dVlt2dt = (1. - 1. / 3.) * (
    # 		(pbc(inVfs, x + 1, y)[0] - 2.0 * C[0] +
    # 		 pbc(inVfs, x - 1, y)[0]) * cddx +
    # 		(pbc(inVfs, x, y + 1)[0] - 2.0 * C[0] +
    # 		 pbc(inVfs, x, y - 1)[0]) * cddy) + (1. / 3.) * 0.5 * (
    # 			 pbc(inVfs, x + 1, y + 1)[0] + pbc(
    # 				 inVfs, x + 1, y - 1)[0] + pbc(inVfs, x - 1, y - 1)[0] +
    # 			 pbc(inVfs, x - 1, y + 1)[0] - 4.0 * C[0]) * (cddx + cddy)
    # 	dVlt2dt *= diffCoef
    
    I_sum = Isi + Ifi + Iso
    dVlt2dt -= I_sum / C_m
    return np.array((dVlt2dt,dFig2dt,dSig2dt),dtype=np.float64)

@njit
def get_time_step (texture, out):
    width  = int(texture.shape[0])
    height = int(texture.shape[1])
    for x in range(width):
        for y in range(height):
            out[x,y] = time_step_at_pixel(texture,x,y)

@njit # or perhaps @jit, which probably won't speed up time_step
def time_step (texture, h, zero_txt):
    dtexture_dt = zero_txt.copy()
    get_time_step(texture, dtexture_dt)
    texture += h * dtexture_dt

#######################################
# Auxiliary functions
#######################################


@njit
def current_at_pixel (inVfs, x, y):#, h):
	print('Hey, are the parameters in current_at_pixel up to date?')
	# define parameters
	width  = int(inVfs.shape[0])
	height = int(inVfs.shape[1])
	ds_x   = 5#18#5#18 #domain size
	ds_y   = 5#18#5#18
	diffCoef = 0.0005
	# diffCoef = 0.001
	C_m = 1.0

	#nonchaos parameters
	# tau_pv = 3.33
	# tau_v1 = 19.6
	# tau_v2 = 1000
	# tau_pw = 667
	# tau_mw = 11
	# tau_d  = 0.42
	# tau_0  = 8.3
	# tau_r  = 50
	# tau_si = 45
	# K      = 10
	# V_sic  = 0.85
	# V_c    = 0.13
	# V_v    = 0.055
	# C_si   = 1.0
	# Uth    = 0.9

	#chaos parameters
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
	I_sum = Isi + Ifi + Iso

	# /*------------------------------------------------------------------------
	#  * Time integration for 0D membrane
	#  *------------------------------------------------------------------------
	#  */
	# dVlt2dt = I_sum / C_m
	# dFig2dt = (1.0 - p) * (1.0 - fig) / tau_mv - p * fig / tau_pv
	# dSig2dt = (1.0 - p) * (1.0 - sig) / tau_mw - p * sig / tau_pw
	#fig += dFig2dt * h
	#sig += dSig2dt * h

	# /*------------------------------------------------------------------------
	#  * ouputing the shader
	#  *------------------------------------------------------------------------
	#  */
	# state  = (vlt,Ifi, Iso, Isi);
	return np.array((vlt,Ifi, Iso, Isi),dtype=np.float64)

def get_tissue_state(texture, out):
	'''out is the 4-channeled np.ndarray saved to.  texture is txt.  each pixel is set to (vlt,Ifi, Iso, Isi).'''
	# assert(4==out.shape[-1])
	for x in range(512):
		for y in range(512):
			out[x,y] = current_at_pixel(texture,x,y)

# @njit
def _blur_at_pixel(inVfs,x,y):
	'''coefficients returned by GaussianMatrix[1] // MatrixForm'''
	outV  = 0.00987648 * pbc1(inVfs,x-1,y+1) + 0.0796275 * pbc1(inVfs,  x,y+1) + 0.00987648 * pbc1(inVfs, x +1, y + 1)
	outV += 0.0796275  * pbc1(inVfs,x  ,y  ) + 0.641984 *  pbc1(inVfs,  x,  y) + 0.0796275  * pbc1(inVfs, x +1, y    )
	outV += 0.00987648 * pbc1(inVfs,x-1,y-1) + 0.0796275 * pbc1(inVfs,  x,y-1) + 0.00987648 * pbc1(inVfs, x +1, y - 1)
	return outV

# @njit
def blur(texture, out):
	for x in range(512):
		for y in range(512):
			out[x,y] = _blur_at_pixel(texture,x,y)

# @njit
def ifilter(texture):
    return texture>0

# @njit
def get_inc(texture,out):
    blur(ifilter(texture),out)


#TODO: move these tests to a proper testing file.
# assert(time_step(gimage ,0.1) is None)
# assert(gimage.any())
# assert(gimage[0,0].dtype    is not None)
