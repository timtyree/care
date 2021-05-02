#!/usr/bin/env python3
import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
# sys.path.append(os.path.dirname(__file__))
# from ..my_initialization import *

# from setuptools import setup, find_packages
# setup(name = 'lib', packages = find_packages())
# from ...lib import *

# sys.path.append(__file__)

# Path hack.
# import sys, os
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.append(os.path.abspath('../..'))

import numpy as np
# from ..utils import *
from numba import njit
# from ..measure import *
from .. import *

def testme():
    pass

def mytest_compute_arclength_values(node_id_lst,xy_values, s_values, contours):
    #first, a test case for the subroutine compute_arclength_values and find_next_tip
    sorted_id_values=np.argsort(node_id_lst)
    remaining_id_lst=list(sorted_id_values)[::-1]
    j=remaining_id_lst.pop()

    #extract values needed for this contour
    point_target=xy_values[j]
    s=s_values[j]
    node_id=node_id_lst[j]
    contour=contours[s][:-1]#exclude last opint because it is a copy of the first

    #compute arclength to next spiral tip
    j_nxt=find_next_tip(remaining_id_lst, s, s_values)
    node_id_nxt=node_id_lst[j_nxt]
    arclength_values=compute_arclength_values(node_id, node_id_nxt, j, j_nxt, contour, xy_values)
    arclength=arclength_values[-1]

    #test for monotonicity of arclen_values
    assert ((np.diff(arclength_values)>0).all())
    print(f"\tarclength_values increased monotonically,")
    return True

def mytest_archlength_support(xy_values, node_id_lst,s_values,contours):
    print(f"\tcomputing all archlengths...")
    #compute all archlengths, with fixed direction
    j_lst,s_lst,archlen_values_lst, j_nxt_lst=compute_arclength_values_for_tips(xy_values, node_id_lst,s_values,contours)
    print(f"\tall archlengths computed!")

    #test that the number of contour nodes equals the number of values recorded minus twice the number of spiral tips
    #compute the total arclength of spiral tips that are paired
    ntips=s_values.shape[0]
    num_av=0
    for av in archlen_values_lst:
        num_av+=av.shape[0]
    n_nodes=0
    for s in list(set(s_lst)):
        contour=contours[s][:-1]
        n_nodes+=contour.shape[0]
    print(f"\t(num_av-2*ntips,n_nodes)={(num_av-2*ntips,n_nodes)}")
    assert(np.isclose(num_av-2*ntips,n_nodes))
    print(f"\tthe number of contour nodes equals the number of values\nrecorded minus twice the number of spiral tips,")

    #test comp_perimeter on the unit square
    contour=np.array([
        [0,0],[0,1],
        [1,1],[1,0]],dtype=np.float)
    assert(np.isclose(comp_perimeter(contour),4))

    #test that the total perimeter of all tips is equal to the sum of the arclengths
    #compute the total perimeter of contours involved
    net_perimeter=0.
    for s in list(set(s_lst)):
        contour=contours[s][:-1]
        net_perimeter+=comp_perimeter(contour)
    print(f'\tnet_perimeter is {net_perimeter} pixels,')

    #compute the total arclength of spiral tips that are paired
    net_archlength=0.
    for av in archlen_values_lst:
        net_archlength+=av[-1]
    print(f'\tnet_archlength is {net_archlength} pixels,')

    error=net_perimeter-net_archlength
    bootol=np.isclose(error,0.,atol=1e-2)#True for first test case
    assert(np.isclose(error,0.,atol=1e-1))
    print(f"\tthe perimeter agrees with the net archlength to at least\none decimal place.\n")
    return True

#####################################################################
#load a mesh from Luo-Rudy that is near termination
# nb_dir=os.path.dirname(os.path.dirname(os.getcwd()))
nb_dir=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
txt_fn=f'{nb_dir}/Data/test_data/ic008.33_t_900.0.npz'
t=900.
print(f'loading test data from {txt_fn}...')
txt_prev=load_buffer(txt_fn)
width,height=txt_prev.shape[:2]
width,height
inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt_prev)
img=inVc[...,0]
dimgdt=dVcdt[...,0]
V_threshold=-50
compute_all_spiral_tips= get_compute_all_spiral_tips(mode='simp',width=width,height=height)
dict_out=compute_all_spiral_tips(t,img,dimgdt,level1=V_threshold,level2=0.)#,width=width,height=height)
x_values=np.array(dict_out['x'])
y_values=np.array(dict_out['y'])
c_values=np.array(dict_out['x'])
print(f"{dict_out['n']} tips are present at time t={int(t)}.")
# fig = ShowDomain(img,dimgdt,x_values,y_values,0.*c_values,V_threshold,t,inch=6,fontsize=16,vmin_img=-85.,vmax_img=35.,
#                  area=25,frameno=1,save_fn=None,save_folder=None,save=False,annotating=False,axis=[0,200,0,200])
# #compute local spiral tip information
# compute_all_spiral_tips= get_compute_all_spiral_tips(mode='full',width=width,height=height)
# dict_local=compute_all_spiral_tips(t, img, dimgdt, V_threshold, 0.)
# print(dict_local.keys())
#compute spiral tip location and topological number
contours_a = find_contours(img,        level = V_threshold)
contours_b = find_contours(dimgdt,     level = 0.)
s1_lst, s2_lst, x_lst, y_lst = contours_to_simple_tips_pbc(contours_a,contours_b,width, height,jump_threshold = 2,size_threshold = 6)
dict_topo={'x':x_lst,'y':y_lst,'s1':s1_lst,'s2':s2_lst}
dict_topo.keys()
#jit compile module to llvm machine code
locate_nearest_point_index = get_locate_nearest_point_index(width=width,height=height)
distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
project_point_2D=get_project_point_2D(width=width,height=height)
subtract_pbc=get_subtract_pbc(width=width,height=height)
comp_perimeter=get_comp_perimeter(width=width,height=height)
fix_node_id=get_fix_node_id(width=width,height=height)
retval=get_arclength_module(width=width,height=height)
locate_node_indices_simple, locate_node_indices, compute_arclength_values, compute_arclength, compute_arclength_values_upto_next, compute_arclength_values_for_tips=retval
#extract values for arclength measurement
contours1=[np.vstack([c[:,1],c[:,0]]).T for c in contours_a]
contours2=[np.vstack([c[:,1],c[:,0]]).T for c in contours_b]
x_values=np.array(dict_topo['x'])
y_values=np.array(dict_topo['y'])
xy_values=np.array(list(zip(dict_topo['x'],dict_topo['y'])))
s1_values=np.array(dict_topo['s1'])
s2_values=np.array(dict_topo['s2'])
#####################################################################
#####################################################################
#TODO(later):njit/optimize what's after ^this
#precompute the locations spiral tips on the contours
node_id1_lst, node_id2_lst=locate_node_indices(xy_values,s1_values,s2_values,contours1,contours2)

print("for the first contour family...")
#select the first of the remaining spiral tips...
node_id_lst=node_id1_lst
s_values=s1_values
contours=contours1
node_id_lst=node_id1_lst

assert ( mytest_compute_arclength_values(node_id_lst,xy_values,s_values,contours) )
assert ( mytest_archlength_support(xy_values, node_id_lst,s_values,contours) )

print("for the second contour family...")
#select the first of the remaining spiral tips...
node_id_lst=node_id2_lst
s_values=s2_values
contours=contours2
node_id_lst=node_id2_lst

assert ( mytest_compute_arclength_values(node_id_lst,xy_values,s_values,contours) )
assert ( mytest_archlength_support(xy_values, node_id_lst,s_values,contours) )
#####################################################################
