#!/bin/bash/env/python3
#TODO: gpu accelerate marching squares with numba.cuda
#TODO: gpu accelerate tip intersections with numba.cuda
import numpy as np
#llvm jit acceleration
from numba import njit

from skimage import measure

#load the libraries
from lib import *
from lib.dist_func import *
from lib.utils_jsonio import *
from lib.operari import *
from lib.get_tips import *
from lib.intersection import *
from lib.minimal_model import *


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

