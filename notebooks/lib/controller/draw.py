from numba import njit
from numba.typed import List
import numpy as np
from ..utils.stack_txt_LR import *

# @njit
def get_semicircle(txt,deg,x0,y0):
    #make the initialization mesh
    img = 0*txt[...,0].copy()
    color_left_of_line(out=img, x0=x0, y0=y0, deg = deg, value=1.)  #make left of line mesh 0 or 1
    color_outside_range(out=img, x0=x0,y0=y0,r=64, val=0.0)  #make circle mesh 0 or 1
    #already done: hadamard product of the two.
    return img

@njit
def color_within_range(x0,y0,r, out, val=1.0, width=512,height=512):
	for x in range(width):
		dx = x-x0
		for y in range(height):
			dy = y-y0
			if np.sqrt(dx**2+dy**2)<=r:
				out[y,x] = val
@njit
def color_outside_range(x0,y0,r, out, val=0.0):
    width  = out.shape[0]
    height = out.shape[1]
    for x in range(width):
        dx = x-x0
        for y in range(height):
            dy = y-y0
            if np.sqrt(dx**2+dy**2)>r:
                out[y,x] = val


@njit
def make_coordinate_textures(txt):
    txtx = txt.copy()
    txty = txt.copy()
    for y in range(txt.shape[0]):
        for x in range(txt.shape[1]):
            txtx[x,y] = x
            txty[x,y] = y
    return txtx, txty

# @njit  #njit crashes rn
def color_left_of_line(out, x0, y0, deg = 45, value=10.):
    width = out.shape[1]
    x0 = int(x0)
    y0 = int(y0)
    for y in range(out.shape[0]):
        l = linear_interpolate_row_to_column(y, x0=x0, y0=y0, deg = deg)
        for x in range(width):
            if x<l:
                out[y,x] = value

@njit
def linear_interpolate_row_to_column(y, x0, y0, deg = 45):
    '''deg is number of degrees semicircle is cut at
    x0,y0 is the center of the circle in pixel coords
    y is the row input coord
    return x'''
    theta = deg/180*np.pi
    dy = y0 - y # img y coords flipped
    dx = np.around(np.tan(theta)*dy)
    x  = x0 + dx
    return int(x)

def get_blank_txt_LR(width,height,V_initial=-85.,Ca_i_initial = 2*10**-4):
    zero_c2=np.zeros(shape=(width,height,2), dtype=np.float64, order='C')
    zero_c6=np.zeros(shape=(width,height,6), dtype=np.float64, order='C')
    #allocate memory to texture
    inVc=zero_c2.copy()
    inVc[...,0]+=V_initial
    inVc[...,1]+=Ca_i_initial
    outVc=inVc.copy()
    inmhjdfx=zero_c6.copy()
    #open the gates that are open at equilibrium at the resting potential (h, j, f)
    inmhjdfx[...,1]=1.
    inmhjdfx[...,2]=1.
    inmhjdfx[...,4]=1.
    outmhjdfx=inmhjdfx.copy()
    dVcdt=zero_c2.copy()
    txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
    return txt

 # @njit
# def linear_interpolate_to_row(y, r0, c0, r1, c1):
#     m = (c1-c0)/(r1-r0)#slope
#     return m*y+r0

#oops this was for dsc291
@njit
def rescale_score(x):
    '''x is float.  maps [0,1] to [-1,1].'''
    return 2.*(x-0.5)
@njit
def rescale_scores(score_list):
    lst = List()
    for x in score_list:
        lst.append(rescale_score(x))
    return lst


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


retval = list(rescale_scores(np.linspace(0,1,10**5)))
print(len(retval))
