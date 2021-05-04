import numpy as np
from ..utils import *
from numba import njit
from ._find_contours import find_contours
from ._find_tips import contours_to_simple_tips_pbc
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

	def locate_node_indices_simple(xy_values,s_values,contours):
		'''locates the segment in either family of contours in which each spiral tip location lives.
		s1,2_values encodes the topological number.
		Example Usage:
		node_id_lst=locate_node_indices(xy_values,s_values,contours)
		'''
		node_id_lst=[]
		ntips=xy_values.shape[0]
		for j in range(ntips):
			point_target = xy_values[j]
			s=s_values[j]
			c=contours[s][:-1]
			#locate
			node_id=locate_nearest_point_index(c,point_target)
			#correct
			node_id=fix_node_id(c,point_target,node_id)
			#record
			node_id_lst.append(node_id)
		return node_id_lst

	# TODO: dev function that interpolates local EP state along the contour
	# def compute_arc_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
	# 	pass
	def compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
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

	def compute_arclength(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
		arclength_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
		return arclength_values[-1]

	def compute_arclength_values_upto_next(j,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst):
		'''
		Example Usage:
		sorted_id_values=np.argsort(node_id_list);remaining_id_lst=list(sorted_id_values)
		arclen_values,j_nxt=compute_arclength_values_upto_next(j,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst)
		'''
		s=s_values[j]
		# point_target=xy_values[j]
		# print(f"(remaining_id_lst, s, s_values)={(remaining_id_lst, s, s_values)}")
		j_nxt=find_next_tip(remaining_id_lst, s, s_values)
		if j_nxt<-1:#if no more tips were found on this contour,
			# enforce the periodic closed contour condition by summing up to the first pt considered.
			# do ^this with j_nxt as the first node_id of the current contour

			# find the smallest j_nxt that has s_values[j_nxt]==s
			i=0;same_s=False
			ntips=s_values.shape[0]
			while (not same_s)&(i<ntips):
				jj=sorted_id_values[i]
				s_tmp=s_values[jj]
				same_s=s_tmp==s
				i+=1
			if same_s:
				j_nxt=jj
			else:
				raise('Warning: edge case for returning to start of loop didn`t work in compute_arclength_values_upto_next')

		# print(f"summing {j}-->{j_nxt} with remaining pts, {(remaining_id_lst)}")
		node_id=node_id_lst[j]
		node_id_nxt=node_id_lst[j_nxt]
		arclen_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
		#don't trigger an extra arclength_values around the whole loop
		# if len(remaining_id_lst)==1:
		# 	remaining_id_lst.pop()
		return arclen_values,j_nxt

	def compute_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours):
		'''Computes the arclengths between points indicated by xy_values along contours
		Supposes that contours is a list of closed contours in 2D.
		Supposes node_id_lst is adjusted by fix_node_id
		Example Usage:
		j_lst,s_lst,archlen_values_lst, j_nxt_lst=compute_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours)

		TODO(later): move computation of node_id_lst to be within the function that generates spiral tip locations using the method that produces s1_values and s2_values.
		'''
		sorted_id_values=np.argsort(node_id_lst)
		remaining_id_lst=list(sorted_id_values)[::-1]
		j_lst=[];s_lst=[];archlen_values_lst=[];j_nxt_lst=[]
		j_nxt=-9999
		while len(remaining_id_lst)>0:
			#if j_nxt is in remaining_id_lst
			if {j_nxt}.issubset(set(remaining_id_lst)):
				#move j_nxt from remaining_id_lst to j
				remaining_id_lst.remove(j_nxt)
				j=j_nxt
			else:
				#pop the earliest remaining tip, possibly from another contour
				j=remaining_id_lst.pop()

			#extract values needed for this contour
			#     point_target=xy_values[j]
			node_id=node_id_lst[j]
			s=s_values[j]
			contour=contours[s][:-1]

			#compute arclength upto sister tip
			arclen_values,j_nxt=compute_arclength_values_upto_next(j,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst)
			#TODO(...a little later): interpolate voltage to node points

			#record outputs
			j_lst.append(j)
			s_lst.append(s)
			archlen_values_lst.append(arclen_values)
			j_nxt_lst.append(j_nxt)
		return j_lst,s_lst,archlen_values_lst, j_nxt_lst

	return locate_node_indices_simple, locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips

def get_comp_dict_topo_simple(width=200.,height=200.,**kwargs):
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

	def update_dict_topo_with_arclen_observations(dict_topo,xy_values,s1_values,s2_values,contours1,contours2):
		'''
		Example Usage:
		update_dict_topo(dict_topo,xy_values,s1_values,s2_values,contours1,contours2)
		'''
		s_values=s2_values
		contours=contours2
		node_id_lst=locate_node_indices_simple(xy_values,s_values,contours)
		j_lst,s_lst,arclen_values_lst,j_nxt_lst=compute_arclength_values_for_tips(xy_values,node_id_lst,s_values,contours)

		#precompute the locations spiral tips on the contours
		# node_id1_lst, node_id2_lst=locate_node_indices(xy_values,s1_values,s2_values,contours1,contours2)
		# # compute arclengths for the first contour family....
		# node_id_lst=node_id1_lst
		# s_values=s1_values
		# contours=contours1
		# node_id_lst=node_id1_lst
		# j_lst,s_lst,archlen_values_lst, j_nxt_lst=compute_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours)
		# j1_lst,s1_lst,j1_nxt_lst=j_lst,s_lst,j_nxt_lst;archlen1_values_lst=list(archlen_values_lst)
		# compute arclengths for the second contour family....
		# s_values=s2_values
		# contours=contours2
		# node_id_lst=node_id2_lst
		# j_lst,s_lst,arclen_values_lst,j_nxt_lst=compute_arclength_values_for_tips(xy_values,node_id_lst,s_values,contours)
		# j2_lst,s2_lst,j2_nxt_lst=j_lst,s_lst,j_nxt_lst;arclen2_values_lst=list(arclen_values_lst)
		# #using second contour family to determine greater/lesser (dVdt=0)
		# j_lst,s_lst,j_nxt_lst=j2_lst,s2_lst,j2_nxt_lst
		# #using first contour family to determine greater/lesser (V=V_threshold)
		# j_lst,s_lst,j_nxt_lst=j1_lst,s1_lst,j1_nxt_lst

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

	def comp_dict_topo_simple(img,dimgdt,t,width=200,height=200,level1=-40,level2=0.,jump_threshold = 2,size_threshold = 6,**kwargs):
		'''
		Example Usage:
		dict_topo=comp_dict_topo_simple(img,dimgdt,t,width=200,height=200,level1=-40,level2=0.,jump_threshold = 2,size_threshold = 6)

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
		xy_values=np.array(list(zip(dict_topo['x'],dict_topo['y'])))
		s1_values=np.array(dict_topo['s1'])
		s2_values=np.array(dict_topo['s2'])
		#update dict_topo with archlength information
		update_dict_topo_with_arclen_observations(dict_topo,xy_values,s1_values,s2_values,contours1,contours2)
		return dict_topo

	return comp_dict_topo_simple
