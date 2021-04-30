import numpy as np
from ..utils import *
from numba import njit
# archlength.py supports the measurement of strings on 2D curves with periodic boundary conditions.
# Programmer: Tim Tyree
# Date: 4.29.2021

def get_arclength_module(width=200.,height=200.):
	'''jit compiles module to llvm machine code.
	Example Usage:
	retval=get_arclength_module(width=200.,height=200.)
	locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips=retval
	'''
	locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	project_point_2D=get_project_point_2D(width=width,height=height)
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	comp_perimeter=get_comp_perimeter(width=width,height=height)
	fix_node_id=get_fix_node_id(width=width,height=height)
	def locate_node_indices(xy_values,s1_values,s2_values,contours1,contours2):
	    '''locates the segment in either family of contours in which each spiral tip location lives.  
	    s1,2_values encodes the topological number.
	    Example Usage:
	    node_id1_lst, node_id2_lst=locate_node_indices(xy_values,s1_values,s2_values,contours1,contours2)
	    '''
	    node_id1_lst=[];node_id2_lst=[];
	    ntips=xy_values.shape[0]
	    for j in range(ntips):
	        point_target = xy_values[j]
	        s1=s1_values[j]
	        s2=s2_values[j]
	        c1=contours1[s1][:-1]
	        c2=contours2[s2][:-1]
	        #locate
	        node_id1=locate_nearest_point_index(c1,point_target)
	        node_id2=locate_nearest_point_index(c2,point_target)
	        #correct
	        node_id1=fix_node_id(c1,point_target,node_id1)
	        node_id2=fix_node_id(c2,point_target,node_id2)
	        #record
	        node_id1_lst.append(node_id1)
	        node_id2_lst.append(node_id2)
	    return node_id1_lst, node_id2_lst


	def compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
	    '''computes the arclength of the contour from j to j_nxt
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
	    assert(l>0)
	    frac=project_point_2D(point=xy_values[j], segment=segment)
	    assert((frac>=0)&(frac<1))
	    arclen+=(1.-frac)*l;arclen_lst.append(arclen)
	    
	    #extract the middle part.
	    if not is_last:
	        segment_array=contour[node_id+1:node_id_nxt]
	    else:
	        Q=contour[node_id+1:]
	        W=contour[:node_id_nxt]
	        segment_array=np.concatenate([Q,W])
	        assert(segment_array.shape[1]==2)
	        
	    #aggregate/append the middle part    
	    n_segments=segment_array.shape[0]
	    for k in range(n_segments):
	        node_start=node_id+1+k
	        segment=get_segment_pbc(node_start,N_nodes,contour)
	        assert(segment.shape==(2,2))
	        l=distance_L2_pbc(segment[1],segment[0])
	        assert(l>0)
	        arclen+=l;arclen_lst.append(arclen)

	    #compute the final frac
	    node_start=node_id_nxt
	    segment=get_segment_pbc(node_start,N_nodes,contour)
	    l=distance_L2_pbc(segment[1],segment[0])
	    assert(l>0)
	    frac=project_point_2D(point=xy_values[j_nxt], segment=segment)
	    assert((frac>=0)&(frac<1))
	    arclen+=frac*l;arclen_lst.append(arclen)
	    
	    #return the activation front arclength parameter, sigma
	    return np.array(arclen_lst)

	def compute_arclength(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
	    arclength_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
	    return arclength_values[-1]

	def compute_arclength_values_upto_next(j,node_id,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst):
	    '''
	    Example Usage:  
	    arclen_values,j_nxt=compute_arclength_values_upto_next(j,node_id,xy_values,s_values,contour,remaining_id_lst,sorted_id_values)
	    '''
	    s=s_values[j]
	    point_target=xy_values[j]
	    node_id_, node_id_nxt, j_nxt=find_next_tip(point_target, node_id, s, remaining_id_lst, sorted_id_values, s_values, node_id_lst)
	    assert(node_id==node_id_)
	    arclen_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
	    return arclen_values,j_nxt

	def compute_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours):
	    '''
	    Example Usage:
	    j_lst,s_lst,archlen_values_lst=compute_arclength_values_for_tips(node_id_lst,s_values,contours)
	    '''
	    sorted_id_values=np.argsort(node_id_lst)
	    remaining_id_lst=list(sorted_id_values)[::-1]
	    j_lst=[];s_lst=[];archlen_values_lst=[]
	    j_nxt=-9999
	    while len(remaining_id_lst)>0:
	        #if j_nxt is in remaining_id_lst
	        if {j_nxt}.issubset(set(remaining_id_lst)):
	            #move j_nxt from remaining_id_lst to j
	            remaining_id_lst.remove(j_nxt)
	            j=j_nxt
	        else:
	            #pop the earliest remaining tip, possibly from another contour
	            i=remaining_id_lst.pop()
	            j=sorted_id_values[i]

	        #extract values needed for this contour
	        #     point_target=xy_values[j]
	        node_id=node_id_lst[j]
	        s=s_values[j]
	        contour=contours[s][:-1]

	        #compute arclength upto sister tip
	        arclen_values,j_nxt=compute_arclength_values_upto_next(j,node_id,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst)
	        #TODO(...a little later): interpolate voltage to node points

	        #record outputs
	        j_lst.append(j)
	        s_lst.append(s)
	        archlen_values_lst.append(arclen_values)
	    return j_lst,s_lst,archlen_values_lst

	return locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips
