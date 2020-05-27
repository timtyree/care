from skimage import measure
from numba import jit, njit
import numpy as np, os
from lib.intersection import *

def enumerate_tips(tips):
    '''returns n_list, x_list, y_list
    gets tips into neat sorted python primitives'''
    n_list = []; x_lst = []; y_lst = []
    if len(tips)==0:
        return None # [],[],[]
    for n,q in enumerate(tips):
        if not (len(q)==0):
            y, x = q
            x = list(x)
            x.sort()
            y = list(y)
            y.sort()
            n_list.append(n)
            x_lst.append(x)
            y_lst.append(y)
    return n_list, x_lst, y_lst

def list_tips(tips):
    return tips_to_list(tips)
def tips_to_list(tips):
    '''returns x_list, y_list
    ets tips into neat sorted python primitives'''
    x_lst = []; y_lst = []
    if len(tips)==0:
        return x_lst, y_lst#None # [],[]
    for q in tips:
        if not (len(q)==0):
            y, x = q
            x = list(x)
            x.sort()
            y = list(y)
            y.sort()
            x_lst.append(x)
            y_lst.append(y)
    return x_lst, y_lst


#for the chaotic parameters
def get_contours(img_nxt,img_inc):
    contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
    contours_inc = measure.find_contours(img_inc, level=0.9)#,fully_connected='low',positive_orientation='low')
    return contours_raw,contours_inc

#tip locating for stable parameters
# img_inc = (img_nxt * ifilter(dtexture_dt[..., 0]))**2  #mask of instantaneously increasing voltages
# img_inc = filters.gaussian(img_inc,sigma=1.)#,truncate=1.0)
# contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
# contours_inc = measure.find_contours(img_inc, level=0.0005)#,fully_connected='low',positive_orientation='low')


# @jit
# def get_contours(img_nxt,img_inc):
#     contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
#     contours_inc = measure.find_contours(img_inc, level=0.0005)#,fully_connected='low',positive_orientation='low')
#     return contours_raw, contours_inc


def get_tips(contours_raw, contours_inc):
    '''returns tips with indices of parent contours'''
    n_list = []; x_lst = []; y_lst = []
    for n1, c1 in enumerate(contours_raw):
        for n2, c2 in enumerate(contours_inc):
            x1, y1 = (c1[:, 0], c1[:, 1])
            x2, y2 = (c2[:, 0], c2[:, 1])
            # tmp = intersection(x1, y1, x2, y2)
            x, y = intersection(x1, y1, x2, y2)
            # if a tip has been detected, save it and it's contour id's
            if len(x)>0:
                s = (n1,n2)
                x = list(x)
                # x.sort()
                y = list(y)
                # y.sort()
                # tmp = (s,x,y)
                # tips.append(tmp)
                n_list.append(s)
                x_lst.append(x)
                y_lst.append(y)
    return n_list, x_lst, y_lst
