import numpy as np
from ..utils import *
from numba import njit
from ._find_contours import find_contours
from ._find_tips import contours_to_simple_tips_pbc
# archlength.py supports full color measurements of strings on 2D curves with periodic boundary conditions.
# Programmer: Tim Tyree
# Date: 5.4.2021

########################################################################
# contour information in full color
########################################################################
def get_compute_arclength_values_full_color(width=200.,height=200.):
    # TODO: breakdown get_arclength_module into its parts
    # TODO: compute arclength positions up to next
    # TODO: return arclength colors as numpy array
    # Hint: contour_values=interpolate_txt_to_contour(contour,width,height,txt)
	def compute_arclength_values_full_color(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
		'''computes the arclength of the contour from j to j_nxt
		Supposes node_id, node_id_nxt went through fix_node_id
		Example Usage:
		arclength_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
		'''
		is_last=node_id_nxt<=node_id
		arclen=0;arclen_lst=[0]
		N_nodes=contour.shape[0]
		#compute the initial frac
		node_start=node_id
		segment=get_segment_pbc(node_start,N_nodes,contour)
		l=distance_L2_pbc(segment[1],segment[0])
		frac=project_point_2D(point=xy_values[j], segment=segment)
		# assert((frac>=0)&(frac<1))
		arclen+=(1.-frac)*l;arclen_lst.append(arclen)

		#extract the middle part.
		if not is_last:
			segment_array=contour[node_id+1:node_id_nxt]
		else:
			Q=contour[node_id+1:]
			W=contour[:node_id_nxt]
			segment_array=np.concatenate([Q,W])

		#aggregate/append the middle part
		n_segments=segment_array.shape[0]
		for k in range(n_segments):
			node_start=node_id+1+k
			segment=get_segment_pbc(node_start,N_nodes,contour)
			# assert(segment.shape==(2,2))
			l=distance_L2_pbc(segment[1],segment[0])
			arclen+=l;arclen_lst.append(arclen)

		#compute the final frac
		node_start=node_id_nxt
		segment=get_segment_pbc(node_start,N_nodes,contour)
		l=distance_L2_pbc(segment[1],segment[0])
		frac=project_point_2D(point=xy_values[j_nxt], segment=segment)
		# assert((frac>=0)&(frac<1))
		arclen+=frac*l;arclen_lst.append(arclen)

		#return the activation front arclength parameter, sigma
		return np.array(arclen_lst)


def get_update_with_arclen_observations(width=200.,height=200.,**kwargs):
	'''
	Example Usage:
	comp_dict_topo_simple=get_comp_dict_topo_simple(width=200.,height=200.)
	'''
	#jit compile arclength module to llvm machine code
	locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	project_point_2D=get_project_point_2D(width=width,height=height)
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	comp_perimeter=get_comp_perimeter(width=width,height=height)
	fix_node_id=get_fix_node_id(width=width,height=height)
	retval=get_arclength_module(width=width,height=height)
	locate_node_indices_simple, locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips=retval

    def update_with_full_color_observations(dict_topo,contours1,contours2):
    	'''
    	Example Usage:
    	'''
		xy_values=np.array(list(zip(dict_topo['x'],dict_topo['y'])))
		s1_values=np.array(dict_topo['s1'])
		s2_values=np.array(dict_topo['s2'])

    	s_values=s2_values
    	contours=contours2
    	node_id_lst=locate_node_indices_simple(xy_values,s_values,contours)
    	j_lst,s_lst,arclen_values_lst,j_nxt_lst=compute_arclength_values_for_tips(xy_values,node_id_lst,s_values,contours)




    	# sort the greater from lesser spiral arms
    	# identify the greater_arclen by sorting either potential front
    	archlen_max_lst=[]
    	archlen_size_lst=[]
    	for a in arclen_values_lst:
    		archlen_max_lst.append(a[-1])
    		archlen_size_lst.append(a.shape[0])
    	greater_i_lst,lesser_i_lst=compare_spiralarm_size(j_lst, j_nxt_lst, archlen_size_lst)

    	#compute greater/lesser sister pid's using set difference ops
    	# print(j_lst, j_nxt_lst)           #two-to-one maps observation basis to particle basis
    	# print(greater_i_lst,lesser_i_lst) #one-to-one maps particle basis to observation basis
    	greater_pid_lst=[];lesser_pid_lst=[]
    	ntips=len(dict_topo['x'])
    	for pid in range(ntips):
    		lesser_pid={j_lst[lesser_i_lst[pid]],j_nxt_lst[lesser_i_lst[pid]]}.difference({pid})
    		greater_pid={j_lst[greater_i_lst[pid]],j_nxt_lst[greater_i_lst[pid]]}.difference({pid})
    		lesser_pid_lst.append(list(lesser_pid)[0])
    		greater_pid_lst.append(list(greater_pid)[0])

    	#map any archlen results to particle values
    	greater_arclen_lst=[];lesser_arclen_lst=[]
    	greater_arclen_values_lst=[];lesser_arclen_values_lst=[]
    	for greater_i,lesser_i in zip(greater_i_lst,lesser_i_lst):
    		#arclengths
    		greater_arclen_lst.append(archlen_max_lst[greater_i])
    		lesser_arclen_lst.append(archlen_max_lst[lesser_i])

    		#sigma
    		greater_arclen_values_lst.append(arclen_values_lst[greater_i])
    		lesser_arclen_values_lst.append(arclen_values_lst[lesser_i])

    		#TODO(later): bilinearly interpolate txt values to contours
    		#TODO(later): voltage values along the greater/lesser arclength

    	# update dict_topo with greater/lesser arclen
    	dict_topo['pid']                   = list(range(ntips))
    	dict_topo['greater_pid']           = greater_pid_lst
    	dict_topo['lesser_pid']            = lesser_pid_lst
    	dict_topo['greater_arclen']        = greater_arclen_lst
    	dict_topo['lesser_arclen']         = lesser_arclen_lst
    	dict_topo['greater_arclen_values'] = greater_arclen_values_lst
    	dict_topo['lesser_arclen_values']  = lesser_arclen_values_lst
    	return True
    return update_dict_contour_with_full_color_observations

def get_comp_dict_topo_full_color(width=200.,height=200.,level1=-40,level2=0.,jump_threshold = 20,size_threshold = 0,**kwargs):
	'''
	Example Usage:
	'''
	#jit compile arclength module to llvm machine code
	locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	project_point_2D=get_project_point_2D(width=width,height=height)
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	comp_perimeter=get_comp_perimeter(width=width,height=height)
	fix_node_id=get_fix_node_id(width=width,height=height)
	retval=get_arclength_module(width=width,height=height)
	locate_node_indices_simple, locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips=retval

	# update_dict_topo_with_arclen_observations=get_update_dict_topo_with_arclen_observations(width=height,height=height,**kwargs)
    update_dict_contour_with_full_color_observations=get_update_dict_topo_with_arclen_observations(width,height,**kwargs)
	def comp_dict_conturs_full_color(img,dimgdt,t,**kwargs):
		'''
		Example Usage:
		dict_topo=comp_dict_topo_simple(img,dimgdt,t)

		TODO(later):njit this whole function... or just translate to c++
		'''
		#compute spiral tip location and topological number
		contours_a = find_contours(img,        level = level1)
		contours_b = find_contours(dimgdt,     level = level2)
		s1_lst, s2_lst, x_lst, y_lst = contours_to_simple_tips_pbc(contours_a,contours_b,width,height,jump_threshold,size_threshold)
	#     print((s1_lst, s2_lst, x_lst, y_lst))
		dict_topo={'x':x_lst,'y':y_lst,'s1':s1_lst,'s2':s2_lst, 't':t}
		#extract values for each arclength measurement
		contours1=[np.vstack([c[:,1],c[:,0]]).T for c in contours_a]
		contours2=[np.vstack([c[:,1],c[:,0]]).T for c in contours_b]
		#     x_values=np.array(dict_topo['x'])
		#     y_values=np.array(dict_topo['y'])
		# xy_values=np.array(list(zip(dict_topo['x'],dict_topo['y'])))
		# s1_values=np.array(dict_topo['s1'])
		# s2_values=np.array(dict_topo['s2'])
		#update dict_topo with archlength information
		update_dict_contour_with_full_color_observations(dict_topo,contours1,contours2)
		contour_values=interpolate_txt_to_contour(contour,width,height,txt)
		return dict_topo

	return comp_dict_contours_full_color
