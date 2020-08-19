import numpy as np
from numba import njit

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

@njit
def pbc1(S,x,y):
	'''S=texture with size 512,512,1
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
    dx, dy   = (1, 1)# (1/512, 1/512) # size of a pixel

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
    tau_mv = (1.0 - q) * tau_v1 + q * tau_v2

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