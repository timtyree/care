import numpy as np
from .dist_func import get_distance_L2_pbc
from ..measure.utils_measure_tips_cpu import get_compute_all_spiral_tips
from numba import njit
from ..measure._find_contours import find_contours
from ..measure._find_tips import contours_to_simple_tips_pbc
# Programmer: Tim Tyree
# Date: 4.29.2021 

def get_tip_locations(inVc,dVcdt,level1=-40.,level2=0.,compute_all_spiral_tips=None):
	'''quick and dirty method to get_tip_locations... slow if compute_all_spiral_tips is None.'''
#     inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
	width,height=inVc.shape[:2]
	if compute_all_spiral_tips is None:
		compute_all_spiral_tips= get_compute_all_spiral_tips(mode='simp',width=width,height=height)
	img=inVc[...,0]
	dimgdt=dVcdt[...,0]
	dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2)#,width=width,height=height)
	return np.array(dict_out['x']),np.array(dict_out['y'])

#DONE: write function that finds the spiral tip in dict_out that is nearest to the given location
def get_find_nearest_tip(width=200,height=200):
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	def find_nearest_tip(x_values,y_values,x,y):
		'''returns the index of the nearest tip.
		returns -9999 upon failure (fix = increase min_dist)
		Example Usage:
		nearest_tip_index=find_nearest_tip(x_values,y_values,x,y)
		'''
		min_dist=9e9
		nearest_tip_index=-9999
		for n,X,Y in enumerate(zip(x_values,y_values)):
			dist=distance_L2_pbc((X,Y),(x,y))
			if dist<min_dist:
				nearest_tip_index=n
				min_dist=dist
		return nearest_tip_index
	return find_nearest_tip

# find_nearest_tip=get_find_nearest_tip(width=200,height=200)


def get_locate_nearest_point_index(width=200,height=200):
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	@njit
	def locate_nearest_point_index(point_array,point_target):
		'''returns the index of the nearest point.
		returns -9999 upon failure (fix = increase min_dist)
		Example Usage:
		locate_nearest_point_index = get_locate_nearest_point_index(width=200,height=200)
		point_array=np.array(c1)
		point_target=np.array((x,y))
		locate_nearest_point_index(point_array,point_target)
		'''
		min_dist=9e9
		nearest_index=-9999
		N=point_array.shape[0]
		for n in range(N):
			dist=distance_L2_pbc(point_array[n],point_target)
			if dist<min_dist:
				nearest_index=n
				min_dist=dist
		return nearest_index
	return locate_nearest_point_index

def find_next_tip(remaining_id_lst, s, s_values):
	"""find the next tip on the same contour, encoded in s_values.
	Supposes remaining_id_lst is a sorted list of spiral tip indices,
	Example Usage:
	j_nxt=find_next_tip(remaining_id_lst, s, s_values)
	"""
	j_nxt=-9999;k=1
	kmax=len(remaining_id_lst)
	while (j_nxt<0)&(k<=kmax):
		jj=remaining_id_lst[-k] #positive integer valued
		s_nxt=s_values[jj]
		if s_nxt==s:
			#the tips are on the same contour
			j_nxt=jj
		else:
			k+=1
	# if k>kmax:
	#     #a circular path is made with the first tip available
	#     j_nxt=0
	return j_nxt
