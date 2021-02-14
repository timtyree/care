#!/bin/bash/env/python3
#TODO: gpu accelerate marching squares with numba.cuda
#TODO: gpu accelerate tip intersections with numba.cuda
import numpy as np
#llvm jit acceleration
from numba import njit

from skimage import measure
from ..model.minimal_model import fetch_get_time_step
# #load the libraries
from .. import *
# from lib import *
# from lib.dist_func import *
# from lib.utils_jsonio import *
# from lib.operari import *
# from lib.get_tips import *
# from lib.intersection import *
# from lib.minimal_model import *

def get_compute_all_spiral_tips(width,height,mode='full',**kwargs):
    '''Example Usage:
    	compute_all_spiral_tips= get_compute_all_spiral_tips(mode='simp',width=width,height=height):
    '''
    if mode == 'full':
        # @njit
        def compute_all_spiral_tips(t,img,dimgdt,level1,level2):
            #compute all spiral tips present
            retval = find_intersections(img,dimgdt,level1,level2)#,theta_threshold=theta_threshold)
            lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy = retval
            return format_spiral_tips(t,img,dimgdt,level1,level2,lst_values_x,lst_values_y,
                lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy,width,height)
    else:
        #simple version
        # @njit
        def compute_all_spiral_tips(t,img,dimgdt,level1,level2):
            #compute all spiral tips present
            retval = find_intersections(img,dimgdt,level1,level2)#,theta_threshold=theta_threshold)
            return format_spiral_tips_simple(t,retval)

    return compute_all_spiral_tips

# @njit('Dict??')
def format_spiral_tips_simple(t,retval):
    lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy = retval
    # x_values = np.array(lst_values_x)
    # y_values = np.array(lst_values_y)
    # EP states given by bilinear interpolation with periodic boundary conditions
    # v_lst    = interpolate_img(x_values,y_values,width,height,img=img)
    # dvdt_lst = interpolate_img(x_values,y_values,width,height,img=dimgdt)
    n_tips = len(lst_values_x)
    dict_out = {
        't': float(t),
        'n': n_tips,
        'x': lst_values_x,
        'y': lst_values_y,
        'grad_ux': lst_values_grad_ux,
        'grad_uy': lst_values_grad_uy,
        'grad_vx': lst_values_grad_vx,
        'grad_vy': lst_values_grad_vy}
        # 'v':v_lst,
        # 'dvdt':dvdt_lst}
    return dict_out


# @njit
def format_spiral_tips(t,img,dimgdt,level1,level2,lst_values_x,lst_values_y,lst_values_theta,
        lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy,width,height):
    x_values = np.array(lst_values_x)
    y_values = np.array(lst_values_y)
    # EP states given by bilinear interpolation with periodic boundary conditions
    v_lst    = interpolate_img(x_values,y_values,width,height,img=img)
    dvdt_lst = interpolate_img(x_values,y_values,width,height,img=dimgdt)

    n_tips = x_values.size
    dict_out = {
        't': float(t),
        'n': int(n_tips),
        'x': list(lst_values_x),
        'y': list(lst_values_y),
        'theta': list(lst_values_theta),
        'grad_ux': list(lst_values_grad_ux),
        'grad_uy': list(lst_values_grad_uy),
        'grad_vx': list(lst_values_grad_vx),
        'grad_vy': list(lst_values_grad_vy),
        'v':v_lst,
        'dvdt':dvdt_lst,
    }
    return dict_out
def fetch_txt_to_tip_dict(width,height,DX=0.025,DY=0.025, **param_dict):
    get_time_step=fetch_get_time_step(width,height,DX=0.025,DY=0.025, **param_dict)
    def txt_to_tip_dict(txt, nanstate, zero_txt, x_coord_mesh, y_coord_mesh,
                        pad, edge_tolerance, atol, tme):
        '''instantaneous method of tip detection'''
        width, height, channel_no = txt.shape
        #calculate discrete flow map
        dtexture_dt = zero_txt.copy()
        get_time_step(txt, dtexture_dt)

        #calculate contours and tips after enforcing pbcs
        img_nxt_unpadded = txt[...,0].copy()
        img_inc_unpadded = dtexture_dt[..., 0].copy()
        img_nxt, img_inc = matrices_to_padded_matrices(img_nxt_unpadded, img_inc_unpadded,pad=pad)
        contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
        contours_inc = measure.find_contours(img_inc, level=0.)
        tips  = get_tips(contours_raw, contours_inc)
        tips_mapped = map_pbc_tips_back(tips=tips, pad=pad, width=width, height=height,
                          edge_tolerance=edge_tolerance, atol = atol)

        n = count_tips(tips_mapped[2])
        #record spiral tip locations
        s1_lst, s2_lst, x_lst, y_lst = tips_mapped
        dict_out = {
                    't': float(tme),
                    'n': int(n),
                    'x': tuple(x_lst),
                    'y': tuple(y_lst),
                    's1': tuple(s1_lst),
                    's2': tuple(s2_lst),
        }
        return dict_out
    return txt_to_tip_dict

def txt_to_tip_dict_with_EP_state(width,height,DX=0.025,DY=0.025, **param_dict):
    get_time_step=fetch_get_time_step(width,height,DX=0.025,DY=0.025, **param_dict)
    def txt_to_tip_dict_with_EP_state(txt, nanstate, zero_txt, x_coord_mesh, y_coord_mesh,
                        pad, edge_tolerance, atol):
        '''instantaneous method of tip detection. with one of three routines for interpolating EP state in use'''
        #calculate discrete flow map
        dtexture_dt = zero_txt.copy()
        get_time_step(txt, dtexture_dt)

        #calculate contours and tips after enforcing pbcs
        img_nxt_unpadded = txt[...,0].copy()
        img_inc_unpadded = dtexture_dt[..., 0].copy()
        img_nxt, img_inc = matrices_to_padded_matrices(img_nxt_unpadded, img_inc_unpadded,pad=pad)
        contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
        contours_inc = measure.find_contours(img_inc, level=0.)
        tips  = get_tips(contours_raw, contours_inc)
        tips_mapped = map_pbc_tips_back(tips=tips, pad=pad, width=width, height=height,
                          edge_tolerance=edge_tolerance, atol = atol)

        #extract local EP field values for each tip
        states_EP = get_states(tips_mapped, txt, pad, nanstate, xcoord_mesh, ycoord_mesh, channel_no = channel_no)
        tips_mapped = add_states(tips_mapped, states_EP)
        n = count_tips(tips_mapped[2])
        #record spiral tip locations
        s1_lst, s2_lst, x_lst, y_lst, states_nearest, states_interpolated_linear, states_interpolated_cubic = tips_mapped
        dict_out = {
                    't': float(tme),
                    'n': int(n),
                    'x': tuple(x_lst),
                    'y': tuple(y_lst),
                    's1': tuple(s1_lst),
                    's2': tuple(s2_lst),
            'states_interpolated_linear': tuple(states_interpolated_linear)
        }
        return dict_out
    return txt_to_tip_dict_with_EP_state
