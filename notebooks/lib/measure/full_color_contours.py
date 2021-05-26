import numpy as np
from ..utils import *
from ..utils.utils_traj import find_jumps
from numba import njit
from ._find_contours import find_contours
from ._find_tips import contours_to_simple_tips_pbc
from .arclength import *
from .. import *
from .interpolate import *
from .curvature import *
# arclength.py supports full color measurements of strings on 2D curves with periodic boundary conditions.
# Programmer: Tim Tyree
# Date: 5.4.2021

########################################################################
# contour information given in full color
########################################################################
def get_compute_arclength_values_full_color(width,height):
	locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
	distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	project_point_2D=get_project_point_2D(width=width,height=height)
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	comp_perimeter=get_comp_perimeter(width=width,height=height)
	fix_node_id=get_fix_node_id(width=width,height=height)

	locate_node_indices=get_locate_node_indices(width,height)
	locate_node_indices_simple=get_locate_node_indices_simple(width,height)

	# TODO: dev function that interpolates local EP state along the contour
	# def compute_arc_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values):
	# 	pass
	compute_arclength_positions=get_compute_arclength_positions(width,height)
	def compute_arclength_values_full_color(txt, xy_values, contour, node_id, node_id_nxt, j, j_nxt):
		'''computes the arclength of the contour from j to j_nxt
		Supposes node_id, node_id_nxt went through fix_node_id
		Example Usage:
		arclength_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
		'''

		arclen_values, contour_xy_values=compute_arclength_positions(xy_values, contour, node_id, node_id_nxt, j, j_nxt)
		width,height=txt.shape[:2]

		contour_color_values=np.array(interpolate_txt_to_contour(contour_xy_values,width,height,txt))
		assert(contour_xy_values.shape[1]==2)
		x_values=contour_xy_values[:,0]
		y_values=contour_xy_values[:,1]
		x_values,y_values=unwrap_contour(x_values,y_values,width,height)

		contour_xy_values_unwrapped=np.stack((x_values,y_values)).T
		assert(contour_xy_values_unwrapped.shape[1]==2)
		# new_points=np.array(comp_interpolated_points(contour_xy_values_unwrapped))
		#slow and noisy
		# dict_curvature=compute_curvature(new_points)
		# #simple and smooth
		# # print(contour_xy_values_unwrapped.shape)
		curvature_values=comp_curvature(contour_xy_values_unwrapped)#,s=2.)
		dict_curvature={'curvature':curvature_values[:-1]}
		# dict_curvature['curvature']=curvature_values
		return arclen_values, contour_color_values, dict_curvature, contour_xy_values

	return compute_arclength_values_full_color

def get_format_dict_contour(width,height,model='LR'):
	# compute_arclength_values_full_color=get_compute_arclength_values_full_color(width,height)
	if model != 'LR':
		raise('model not yet implemented!')
	def format_dict_contour(arclen_values,  contour_color_values, dict_curvature=None):
		'''contour information given in full color'''
		# TODO: format contour as DataFrame using the activation front arclength parameter, sigma
		assert contour_color_values.shape[1]==18 #redundant with model kwarg
		c_values=contour_color_values.T
		dict_contour={
			'sigma':arclen_values,
			'V':c_values[0].copy(),
			'dVdt':c_values[-2].copy(),
			'Ca':c_values[1].copy(),
			'dCadt':c_values[-1].copy()
		}
		if dict_curvature is not None:
			dict_contour.update(dict_curvature)
		return dict_contour
	return format_dict_contour

def get_update_with_full_color_observations(width,height,**kwargs):
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
	compute_colored_arclength_values_for_tips=get_compute_colored_arclength_values_for_tips(width,height)
	compute_arclength_values_full_color=get_compute_arclength_values_full_color(width,height)
	format_dict_contour=get_format_dict_contour(width,height)
	def update_with_full_color_observations(dict_topo,contours1,contours2,txt):
		'''
		Example Usage:
		'''
		xy_values=np.array(list(zip(dict_topo['x'],dict_topo['y'])))
		s1_values=np.array(dict_topo['s1'])
		s2_values=np.array(dict_topo['s2'])
		s_values=s1_values
		contours=contours1
		node_id_lst=locate_node_indices_simple(xy_values,s_values,contours)
		# j_lst,s_lst,arclen_values_lst,j_nxt_lst=compute_arclength_values_for_tips(xy_values,node_id_lst,s_values,contours)
		retval=compute_colored_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours,txt)
		j_lst,s_lst,arclen_values_lst, j_nxt_lst,dict_contour_lst,mean_V_lst,mean_dVdt_lst,mean_Ca_lst,mean_dCadt_lst,mean_curvature_lst,xy_values_lst=retval
		#TODO(later): output full contour_color_values_lst
		#TODO: associate contour with lesser/greater tip index.  then, give contour an index number
		# sort the greater from lesser spiral arms
		# identify the greater_arclen by sorting either potential front
		arclen_max_lst=[]
		arclen_size_lst=[]
		for a in arclen_values_lst:
			arclen_max_lst.append(a[-1])
			arclen_size_lst.append(a.shape[0])

		greater_i_lst,lesser_i_lst=compare_spiralarm_size(j_lst, j_nxt_lst, arclen_size_lst)

		#sort greater/lesser spiral arms using full color information
		# greater_i_lst,lesser_i_lst=compare_spiralarm_voltage(j_lst, j_nxt_lst, avgVoltage_lst=mean_dVdt_lst)

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
		#map any arclen results to particle values
		greater_arclen_lst=[];lesser_arclen_lst=[]
		greater_arclen_values_lst=[];lesser_arclen_values_lst=[]
		greater_mean_V_lst=[];lesser_mean_V_lst=[]
		greater_V_values_lst=[];lesser_V_values_lst=[]
		greater_mean_curvature_lst=[];lesser_mean_curvature_lst=[]
		greater_curvature_lst=[];lesser_curvature_lst=[]
		greater_curvature_values_lst=[];lesser_curvature_values_lst=[]
		greater_xy_values_lst=[];lesser_xy_values_lst=[];
		for greater_i,lesser_i in zip(greater_i_lst,lesser_i_lst):
			#arclengths
			greater_arclen_lst.append(arclen_max_lst[greater_i])
			lesser_arclen_lst.append(arclen_max_lst[lesser_i])

			#sigma
			greater_arclen_values_lst.append(arclen_values_lst[greater_i])
			lesser_arclen_values_lst.append(arclen_values_lst[lesser_i])

			#voltage
			greater_mean_V_lst.append(mean_V_lst[greater_i])
			lesser_mean_V_lst.append(mean_V_lst[lesser_i])

			#voltage values
			greater_V_values_lst.append(dict_contour_lst[greater_i]['V'])
			lesser_V_values_lst.append(dict_contour_lst[lesser_i]['V'])

			#curvature
			greater_mean_curvature_lst.append(mean_curvature_lst[greater_i])
			lesser_mean_curvature_lst.append(mean_curvature_lst[lesser_i])

			#curvature values
			greater_curvature_values_lst.append(dict_contour_lst[greater_i]['curvature'])
			lesser_curvature_values_lst.append(dict_contour_lst[lesser_i]['curvature'])

			#xy values
			greater_xy_values_lst.append(xy_values_lst[greater_i])
			lesser_xy_values_lst.append(xy_values_lst[lesser_i])



			#TODO(later): add the following information to dict_topo
			# dict_contour_lst,mean_V_lst,mean_dVdt_lst,mean_Ca_lst,mean_dCadt_lst,mean_curvature_lst
			#TODO(later, earlier in code): map between particles (pid) and contours (cid)
		# update dict_topo with greater/lesser arclen
		dict_topo['pid']                   = list(range(ntips))
		dict_topo['greater_pid']           = greater_pid_lst
		dict_topo['lesser_pid']            = lesser_pid_lst
		dict_topo['greater_arclen']        = greater_arclen_lst
		dict_topo['lesser_arclen']         = lesser_arclen_lst
		dict_topo['greater_arclen_values'] = greater_arclen_values_lst
		dict_topo['lesser_arclen_values']  = lesser_arclen_values_lst
		dict_topo['greater_arclen']        = greater_arclen_lst
		dict_topo['lesser_arclen']         = lesser_arclen_lst

		dict_topo['greater_mean_V']        = greater_mean_V_lst
		dict_topo['lesser_mean_V']         = lesser_mean_V_lst

		dict_topo['greater_mean_curvature']        = greater_mean_curvature_lst
		dict_topo['lesser_mean_curvature']         = lesser_mean_curvature_lst

		dict_topo['greater_xy_values']        = greater_xy_values_lst
		dict_topo['lesser_xy_values']         = lesser_xy_values_lst

		dict_topo['greater_V_values']        = greater_V_values_lst
		dict_topo['lesser_V_values']         = lesser_V_values_lst

		dict_topo['greater_curvature_values']        = greater_curvature_values_lst
		dict_topo['lesser_curvature_values']         = lesser_curvature_values_lst

		return True
	return update_with_full_color_observations

def get_comp_dict_topo_full_color(width,height,level1=-50,level2=0.,jump_threshold=100,size_threshold=0,ds=5.,**kwargs):
	'''
	Example Usage:
	'''
	# #jit compile arclength module to llvm machine code
	# locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
	# distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
	# project_point_2D=get_project_point_2D(width=width,height=height)
	# subtract_pbc=get_subtract_pbc(width=width,height=height)
	# comp_perimeter=get_comp_perimeter(width=width,height=height)
	# fix_node_id=get_fix_node_id(width=width,height=height)
	retval=get_arclength_module(width=width,height=height)
	locate_node_indices_simple, locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips=retval
	# update_dict_topo_with_arclen_observations=get_update_dict_topo_with_arclen_observations(width=height,height=height,**kwargs)
	update_with_full_color_observations=get_update_with_full_color_observations(width,height,**kwargs)
	compute_colored_arclength_values_for_tips=get_compute_colored_arclength_values_for_tips(width,height,**kwargs)
	scale=ds/width  #cm per pixel
	def comp_dict_topo_full_color(img,dimgdt,t,txt,**kwargs):
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
		#update dict_topo with arclength information
		update_with_full_color_observations(dict_topo,contours1,contours2,txt)
		# ntips=dict_topo[]
		# # scale all arclen values to centimeters

		# dict_keys(['x', 'y', 's1', 's2', 't', 'pid', 'greater_pid', 'lesser_pid',
		# 'greater_arclen', 'lesser_arclen', 'greater_arclen_values',
		# 'lesser_arclen_values', 'greater_mean_V', 'lesser_mean_V',
		# 'greater_mean_curvature', 'lesser_mean_curvature', 'greater_xy_values',
		# 'lesser_xy_values', 'greater_V_values', 'lesser_V_values',
		# 'greater_curvature_values', 'lesser_curvature_values'])
		dict_topo['lesser_arclen']=[v*scale for v in dict_topo['lesser_arclen']]
		dict_topo['greater_arclen']=[v*scale for v in dict_topo['greater_arclen']]

		dict_topo['lesser_arclen_values']=[v*scale for v in dict_topo['lesser_arclen_values']]
		dict_topo['greater_arclen_values']=[v*scale for v in dict_topo['greater_arclen_values']]

		dict_topo['lesser_mean_curvature']=[v/scale for v in dict_topo['lesser_mean_curvature']]
		dict_topo['greater_mean_curvature']=[v/scale for v in dict_topo['greater_mean_curvature']]

		dict_topo['lesser_curvature_values']=[v/scale for v in dict_topo['lesser_curvature_values']]
		dict_topo['greater_curvature_values']=[v/scale for v in dict_topo['greater_curvature_values']]
		# contour_values=interpolate_txt_to_contour(contour,width,height,txt)
		return dict_topo
	return comp_dict_topo_full_color

def get_compute_colored_arclength_values_for_tips(width,height,**kwargs):
	compute_arclength_values_upto_next=get_compute_arclength_values_upto_next(width,height)
	compute_arclength_values_full_color=get_compute_arclength_values_full_color(width,height)
	format_dict_contour=get_format_dict_contour(width,height)

	def compute_colored_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours,txt):
		'''Computes the arclengths between points indicated by xy_values along contours
		Supposes that contours is a list of closed contours in 2D.
		Supposes node_id_lst is adjusted by fix_node_id
		Example Usage:
		retval=compute_colored_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours,txt)
		j_lst,s_lst,arclen_values_lst, j_nxt_lst,dict_contour_lst,mean_V_lst,mean_dVdt_lst,mean_Ca_lst,mean_dCadt_lst,mean_curvature_lst=retval

		TODO(later): move computation of node_id_lst to be within the function that generates spiral tip locations using the method that produces s1_values and s2_values.
		'''
		sorted_id_values=np.argsort(node_id_lst)
		remaining_id_lst=list(sorted_id_values)[::-1]
		j_lst=[];s_lst=[];arclen_values_lst=[];j_nxt_lst=[]
		dict_contour_lst=[];mean_V_lst=[];mean_dVdt_lst=[];mean_Ca_lst=[];mean_dCadt_lst=[];mean_curvature_lst=[]
		xy_values_lst=[]
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

			#find next tip in the same contour
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
			node_id_nxt=node_id_lst[j_nxt]

			#compute arclength upto sister tip
			# arclen_values,j_nxt=compute_arclength_values_upto_next(j,xy_values,s_values,contour,remaining_id_lst,sorted_id_values,node_id_lst)
			arclen_values,  contour_color_values, dict_curvature, contour_xy_values=compute_arclength_values_full_color(txt, xy_values, contour, node_id, node_id_nxt, j, j_nxt)
			dict_contour=format_dict_contour(arclen_values,  contour_color_values, dict_curvature)

			mean_V = np.mean(dict_contour['V'])
			mean_dVdt = np.mean(dict_contour['dVdt'])
			mean_Ca = np.mean(dict_contour['Ca'])
			mean_dCadt = np.mean(dict_contour['dCadt'])
			mean_curvature = np.mean(dict_contour['curvature'])

			#record outputs
			j_lst.append(j)
			s_lst.append(s)
			arclen_values_lst.append(arclen_values)

			j_nxt_lst.append(j_nxt)

			#record colors
			dict_contour_lst.append(dict_contour)
			mean_V_lst.append(mean_V)
			mean_dVdt_lst.append(mean_dVdt)
			mean_Ca_lst.append(mean_Ca)
			mean_dCadt_lst.append(mean_dCadt)
			mean_curvature_lst.append(mean_curvature)
			xy_values_lst.append(contour_xy_values)

		return j_lst,s_lst,arclen_values_lst, j_nxt_lst,dict_contour_lst,mean_V_lst,mean_dVdt_lst,mean_Ca_lst,mean_dCadt_lst,mean_curvature_lst,xy_values_lst
	return compute_colored_arclength_values_for_tips


# measure_subkernel
def find_tips_with_pbc_knots_full_color(contours1, contours2, s1in_lst, s2in_lst, txt):
	'''returns tips with indices of parent contours.
	contours1 and contours2 are a lists of numpy arrays.
	each such numpy array is an Nx2 array representing a contiguous contour (i.e. continuous in local xy coordinates).
	Example Usage:
	retval=find_tips_with_pbc_knots_full_color(contours1, contours2, s1in_lst, s2in_lst, txt)
	s1_list, s2_list, x_lst, y_lst, txt_lst=retval

	#TODO: call ^this function with contours_to_simple_tips_pbc_full_color
	'''
	xshift=0.5;yshift=0.5
	width,height=txt.shape[:2]
	s1_list = []; s2_list = []; x_lst = []; y_lst = []; txt_lst=[]
	for n1, contour1_lst in zip(s1in_lst,contours1):
		for c1 in contour1_lst:
			for n2, contour2_lst in zip(s2in_lst,contours2):
				for c2 in contour2_lst:
					x1, y1 = (c1[:, 0], c1[:, 1])
					x2, y2 = (c2[:, 0], c2[:, 1])
					#intersection is not the cause of the slowdown... it's probably all ^that pythonic voodoo...
					yl, xl = intersection(x1, y1, x2, y2)
					for x,y in zip(xl,yl):
						s1_list.append(n1)
						s2_list.append(n2)
						x_lst.append(x)
						y_lst.append(y)
						#TODO: locally interpolate txt
						txt_lst.append(bilinear_interpolate_channel(x+xshift, y+yshift, width, height, txt))
	#DONE: sort####compute node_id_values=?? using contour1_len_values and locally determined values in the sublist
	sorted_lst=sorted(zip(x_lst,y_lst,s1_lst,s2_lst, txt_lst))
	sorted_values=np.array(sorted_lst)
	in_to_sorted_values=np.argsort(sorted_values)
	x_lst, y_lst, s1_lst, s2_lst=in_to_sorted_values[sorted_values]
	return s1_list, s2_list, x_lst, y_lst, txt_lst
