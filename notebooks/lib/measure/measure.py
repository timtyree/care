from ._utils_find_tips import *
from ._find_tips import *
from numba import njit
import numpy as np
from ._utils_find_tips import reduce_tips
# from ..operari import count_tips

@njit
def __pbc_1d(x_in,width):
	if np.greater_equal(x_in , width):
		return int(x_in-width)
	elif np.less(x_in, 0):
		return int(x_in+width)
	return int(x_in)

@njit
def bilinear_interpolate(x,y,width,height,txt):
	r0 = np.floor(y)
	r1 = np.ceil(y)
	frac_r = y-r0
	if r0==r1:
		r1+=1
	c0 = np.floor(x)
	c1 = np.ceil(x)
	frac_c = x-c0
	if c0==c1:
		c1+=1
	c0 = __pbc_1d(c0,width)
	c1 = __pbc_1d(c1,width)
	r0 = __pbc_1d(r0,height)
	r1 = __pbc_1d(r1,height)
	a11 = txt[r0,c0]
	a21 = txt[r1,c0]-a11
	a12 = txt[r0,c1]-a11
	a22 = txt[r1,c1]-a11-a12-a21
	states_bilinear = a11 + a21*frac_c + a12*frac_r + a22*frac_c*frac_r
	v,f,s = states_bilinear
	return list(states_bilinear)

@njit
def interpolate_states(x_values,y_values,width,height,txt):
	v_lst = []; f_lst = []; s_lst = []; 
	for x,y in zip(x_values,y_values):
		v,f,s = bilinear_interpolate(x,y,width,height,txt)
		v_lst.append(v)
		f_lst.append(f)
		s_lst.append(s)
	return v_lst, f_lst, s_lst

def measure_system(contours1, contours2, width, height, texture, jump_threshold = 2, size_threshold = 6, pad=1, decimals=11):
	'''returns s1_list, s2_list, x_lst, y_lst, v_lst, f_lst, s_lst
	Nota Bene: The vast majority of the runtime of measure_system results from find_tips due to the list of lists datastructure necessary to encode topological information.  
	# Actual acceleration of find_tips using C or LLVM is not straightforward due to memory allocation issues...
	# Speedup is possible, but topological information would be destroyed.'''
	contour1_lst_lst, contour2_lst_lst = preprocess_contours(contours1, contours2, width, height, jump_threshold = jump_threshold, size_threshold = size_threshold)
	s1_list, s2_list, x_lst, y_lst = find_tips(contour1_lst_lst, contour2_lst_lst)
	s1_list, s2_list, x_lst, y_lst = reduce_tips(s1_list, s2_list, x_lst, y_lst, width, height, pad=pad, decimals=decimals)
	v_lst, f_lst, s_lst = interpolate_states(x_lst,y_lst,width,height,texture)
	return s1_list, s2_list, x_lst, y_lst, v_lst, f_lst, s_lst
	# dict_out = {
	# 		't': float(t),
	# 		'n': int(n_tips),
	# 		'x': tuple(x_values),
	# 		'y': tuple(y_values),
	# 		'n1': tuple(s1_list),
	# 		'n2': tuple(s2_list),
	# 		'v':v_lst,
	# 		'f':f_lst,
	# 		's':s_lst,
	# 	}
	# sys.getsizeof(dict_out) one time step fills about 360 [bytes or bits?]
	# return dict_out
