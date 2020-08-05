from skimage import measure
from numba import jit, njit
from numba.typed import List
import numpy as np, os
from lib.intersection import *
from scipy.interpolate import interp2d


# @njit
def get_tips(contours_raw, contours_inc):
    '''returns tips with indices of parent contours'''
    n_list = []; x_lst = []; y_lst = []
    for n1, c1 in enumerate(contours_raw):
        for n2, c2 in enumerate(contours_inc):
            x1, y1 = (c1[:, 0], c1[:, 1])
            x2, y2 = (c2[:, 0], c2[:, 1])
            x, y = intersection(x1, y1, x2, y2)
            if len(x)>0:
                s = (n1,n2)
                x = list(x)
                y = list(y)
                n_list.append(s)
                x_lst.append(x)
                y_lst.append(y)
    return n_list, x_lst, y_lst


# @njit#(cache=True)#, nogil = True)
# def get_tips(contours_a,contours_b):
#     '''returns tips with indices of parent contours returned as the nested list, n_list.
#     tuple(contours_a),tuple(contours_b) are each tuples of m-by-2 np.ndarrays. m is any positive int.
#     each member is a 1D line.  
    
#     get_tips returns all intersections of 
#     contours_a with contours_b.  
#     will throw a TypingError exception if either input tuple is empty.
    
#     if you get a nonsingular matrix error, make sure that you`re not comparing a contour to itself.'''
#     n_list = List(); x_list = List(); y_list = List();
#     ncr = len(contours_a); nci = len(contours_b)
#     for n1 in prange(ncr):
#         for n2 in prange(nci):
# #     for n1, c1 in enumerate(contours_a):
# #         for n2, c2 in enumerate(contours_b):
#             c1 = contours_a[n1]
#             c2 = contours_b[n2]
#             x1 = c1[:, 0]
#             y1 = c1[:, 1]
#             x2 = c2[:, 0]
#             y2 = c2[:, 1]
#             x,y = intersection(x1, y1, x2, y2)
#             if len(x)>0:
#                 s = (n1,n2)
#                 xl = list(x)
#                 yl = list(y)
#                 n_list.append(s)
#                 x_list.append(xl)
#                 y_list.append(yl)
#     return n_list, x_list, y_list

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


#deprecated - needs parameters
# def get_contours(img_nxt,img_inc):
#     contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
#     contours_inc = measure.find_contours(img_inc, level=0.9,fully_connected='low',positive_orientation='low')
#     return contours_raw,contours_inc

#tip locating for stable parameters
# img_inc = (img_nxt * ifilter(dtexture_dt[..., 0]))**2  #mask of instantaneously increasing voltages
# img_inc = filters.gaussian(img_inc,sigma=2., mode='wrap')
# contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
# contours_inc = measure.find_contours(img_inc, level=0.0005)#,fully_connected='low',positive_orientation='low')


# @jit
# def get_contours(img_nxt,img_inc):
#     contours_raw = measure.find_contours(img_nxt, level=0.5,fully_connected='low',positive_orientation='low')
#     contours_inc = measure.find_contours(img_inc, level=0.0005)#,fully_connected='low',positive_orientation='low')
#     return contours_raw, contours_inc


# def get_tips(contours_raw, contours_inc):
#     '''returns tips with indices of parent contours'''
#     n_list = []; x_lst = []; y_lst = []
#     for n1, c1 in enumerate(contours_raw):
#         for n2, c2 in enumerate(contours_inc):
#             x1, y1 = (c1[:, 0], c1[:, 1])
#             x2, y2 = (c2[:, 0], c2[:, 1])
#             # tmp = intersection(x1, y1, x2, y2)
#             x, y = intersection(x1, y1, x2, y2)
#             # if a tip has been detected, save it and its contour ids
#             if len(x)>0:
#                 s = (n1,n2)
#                 x = list(x)
#                 # x.sort()
#                 y = list(y)
#                 # y.sort()
#                 # tmp = (s,x,y)
#                 # tips.append(tmp)
#                 n_list.append(s)
#                 x_lst.append(x)
#                 y_lst.append(y)
#     return n_list, x_lst, y_lst

def my_numba_list_to_python_list(numba_lst):
    normal_list = []
    for lst in numba_lst:
        normal_list.append(list(lst))
    return normal_list

@njit
def unpad_xy_position (position, pad_x, width, rejection_distance_x, 
                       pad_y, height, rejection_distance_y):
    x = unpad(X=position[0], pad=pad_x, width=width, rejection_distance=rejection_distance_x)
    y = unpad(X=position[1], pad=pad_y, width=height, rejection_distance=rejection_distance_y)
    return x,y    

@njit
def unpad(X, pad, width, rejection_distance):
    '''unpads 1 coordinate x or y for the padding: 
    [0... pad | pad ... width + pad | width + pad ... width + 2 * pad]
    return -9999 if X is within rejection_distance of the edge,
    return X if X is in [pad ... width + pad], which is if X is in the unpadded frame, which has width = width
    else return X reflected onto the unpadded frame'''
    P  = rejection_distance
    X -= pad
    if X < -pad+P:
        X = -9999 # throw out X later
    elif X < 0:
        X += width
    if X > width+pad-P:
        X = -9999 # throw out X later
    elif X >= width:
        X -= width
    return X


# @njit
def textures_to_padded_textures(txt,dtexture_dt, pad):
    '''large pad allows knots to be recorded right.
    consider pad = int(512/2), edge_tolerance = int(512/4)'''
    width, height = txt.shape[:2]
    # padded_width = 512 + pad #pixels
    padded_txt     = np.pad(array = txt[...,0],        pad_width = pad, mode = 'wrap')
    dpadded_txt_dt = np.pad(array = dtexture_dt[...,0], pad_width = pad, mode = 'wrap')
    return padded_txt, dpadded_txt_dt

def matrices_to_padded_matrices(txt,dtexture_dt, pad):
    '''txt and dtexture_dt are rank two tensors. i.e. the channel_no is 1.
    large pad allows knots to be recorded right.
    '''
    # width, height = txt.shape[:2]
    # padded_width = 512 + pad #pixels
    padded_txt     = np.pad(array = txt,        pad_width = pad, mode = 'wrap')
    dpadded_txt_dt = np.pad(array = dtexture_dt, pad_width = pad, mode = 'wrap')
    return padded_txt, dpadded_txt_dt

# #informal test for ^that
# padded_txt     = np.pad(array = txt,        pad_width = pad, mode = 'wrap')
# print(txt[0,0])
# print(padded_txt[...,2:5][pad,pad])

# @njit
def pad_matrix(mat, pad):
    '''large pad allows knots to be recorded right.
    consider pad = int(512/2), edge_tolerance = int(512/4)'''
    width, height = mat.shape[:2]
    # padded_width = 512 + pad #pixels
    padded_mat = np.pad(array = mat, pad_width = pad, mode = 'wrap')
    return padded_mat[...,2:5]

# @njit
def pad_texture(txt, pad):
    '''large pad allows knots to be recorded right.
    consider pad = int(512/2), edge_tolerance = int(512/4)'''
    width, height = txt.shape[:2]
    # padded_width = 512 + pad #pixels
    padded_txta     = np.pad(array = txt[...,0],        pad_width = pad, mode = 'wrap')
    padded_txtb     = np.pad(array = txt[...,1],        pad_width = pad, mode = 'wrap')
    padded_txtc     = np.pad(array = txt[...,2],        pad_width = pad, mode = 'wrap')
    # dpadded_txt_dt = np.pad(array = dtexture_dt[...,0], pad_width = pad, mode = 'wrap')
    return np.array([padded_txta,padded_txtb,padded_txtc]).T

def map_pbc_tips_back(tips, pad, width, height, edge_tolerance, atol = 1e-11):
    '''width and height are from the shape of the unpadded buffer.
    TODO: get intersection to be njit compiled, then njit map_pbc_tips_back, 
    for which I'll need to return to using numba.typed.List() instead of [].'''
    atol_squared = atol**2
    min_dist_squared_init = width**2
    s_tips, x_tips, y_tips = tips
    s1_mapped_lst = []; s2_mapped_lst = [];
    x_mapped_lst  = []; y_mapped_lst  = [];
    #     s1_mapped_lst = List(); s2_mapped_lst = List();
    #     x_mapped_lst  = List(); y_mapped_lst  = List(); 
    for n, x in enumerate(x_tips):
        y = y_tips[n]; s = s_tips[n]
        S1, S2 = s_tips[n]
        y = y_tips[n]
        for X, Y in zip(x, y):
            X = unpad(X=X, pad=pad, width=width , rejection_distance=edge_tolerance)
            if not (X == -9999):
                Y = unpad(X=Y, pad=pad, width=height, rejection_distance=edge_tolerance)
                if not (Y == -9999):

                    # find the index and distance to the nearest tip already on the mapped_lsts
                    min_dist_squared = min_dist_squared_init; min_index = -1
                    for j0, (x0,y0) in enumerate(zip(x_mapped_lst,y_mapped_lst)):
                        # compute the distance between x0,y0 and X,Y
                        dist_squared = (X-x0)**2+(Y-y0)**2
                        # if ^that distance is the smallest, update min_dist with it
                        if dist_squared < min_dist_squared:
                            min_dist_squared = dist_squared
                            min_index = j0

                    #if this new tip is sufficiently far from all other recorded tips,
                    if min_dist_squared >= atol:
                        # then append the entry to all four lists        
                        x_mapped_lst.append(X)
                        y_mapped_lst.append(Y)
                        lst_S1 = []#List()
                        lst_S1.append(S1)
                        lst_S2 = []#List()
                        lst_S2.append(S2)
                        s1_mapped_lst.append(lst_S1)
                        s2_mapped_lst.append(lst_S2)
                    else:
                        #just append to the previous entry in the s1 and s2 lists if the contour isn't already there
                        s1_mapped_lst[min_index].append(S1)
                        s2_mapped_lst[min_index].append(S2)
    return s1_mapped_lst, s2_mapped_lst, x_mapped_lst, y_mapped_lst


#########################################################################
# Interpolating Electrophysiological state values to spiral tip locations
#########################################################################
def get_state_nearest(x, y, txt):
    '''nearest local texture values, ignore any index errors and/or periodic boundary conditions'''
    xint = np.round(x).astype(dtype=int)
    yint = np.round(y).astype(dtype=int)
    try:
        state_nearest = list(txt[xint,yint])
    except IndexError:
        state_nearest = nanstate
    return state_nearest    

#for get_state_interpolated
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore", category=RuntimeWarning, lineno=0, append=False)
#TODO: restrict ^this warning filter to onlyt get_state_interpolated

def get_state_interpolated(x, y, txt, nanstate, xcoord_mesh, ycoord_mesh,
                          channel_no = 3, rad = 0.5, kind='linear'):
    '''linear interpolation of local texture values to subpixel precision
    using 2D linear interpolation with scipy.interpolate.interp2d.
    channel_no must be len(nanstate). 
    for channel_no = 3, use nanstate = [np.nan,np.nan,np.nan].
    rad = the pixel radius considered in interpolation.
    kind can be "linear" or "cubic".  
    if kind="cubic", then set rad = 3.5.'''
    state_interpolated = nanstate #.copy() if you change nanstate to a numpy array
    try:
        xlo = np.round(x-rad).astype(dtype=int)
        ylo = np.round(y-rad).astype(dtype=int)
        xhi = np.round(x+rad).astype(dtype=int)
        yhi = np.round(y+rad).astype(dtype=int)
        yloc = ycoord_mesh[ylo:yhi+1,xlo:xhi+1].flatten().copy()
        xloc = xcoord_mesh[ylo:yhi+1,xlo:xhi+1].flatten().copy()
        local_values = txt[ylo:yhi+1,xlo:xhi+1]

        interp_foo = lambda x,y,zloc: interp2d(yloc,xloc,zloc,kind=kind)(y,x)
        for c in range(channel_no):
            zloc = local_values[...,c].flatten().copy()
            state_interpolated[c] = float(interp_foo(x,y,zloc))
    except IndexError:
        pass
    except RuntimeWarning:
        pass
    return state_interpolated
# ###############
# # Example Usage 
# ###############
# #Caution! : check whether spiral tips are recorded as 'x': x coordinate or 'x': y coordinate

# #precompute the following the __padded__ coordinates
# xcoord_mesh, ycoord_mesh = np.meshgrid(np.arange(0,200),np.arange(0,200))

# x = 169.75099760896785
# y = 68.05364536542943

# nanstate = [np.nan,np.nan,np.nan]
# txt = np.stack([texture,texture,texture]).T
# print(
#     get_state_nearest(x,y,txt)
#     )
# print ( 
#     get_state_interpolated(x, y, txt.astype('float32'), nanstate, xcoord_mesh, ycoord_mesh,
#                           channel_no = 3, rad = 3.5, kind='cubic') 
#       )
# print ( 
#     get_state_interpolated(x, y, txt.astype('float32'), nanstate, xcoord_mesh, ycoord_mesh,
#                           channel_no = 3, rad = 0.5, kind='linear') 
#       )


def get_states(tips_mapped, txt, pad,
              nanstate, xcoord_mesh, ycoord_mesh, channel_no = 3):
    '''iterates through x_locations and y_locations contained in tips_mapped and returns the electrophysiological states'''
    # tips_mapped gives tip locations using the correct image pixel coordinates, here.
    padded_txt  = pad_matrix(txt, pad)
    y_locations = np.array(tips_mapped[2]) + pad
    x_locations = np.array(tips_mapped[3]) + pad

    states_nearest = states_interpolated_linear = states_interpolated_cubic = [];
    for x,y in zip(x_locations,y_locations):
        state_nearest = get_state_nearest(x,y,txt=padded_txt)
        state_interpolated_linear = get_state_interpolated(x, y, padded_txt, nanstate, xcoord_mesh, ycoord_mesh,
                              channel_no = channel_no, rad = 0.5, kind='linear') 
        state_interpolated_cubic = get_state_interpolated(x, y, padded_txt, nanstate, xcoord_mesh, ycoord_mesh,
                              channel_no = channel_no, rad = 3.5, kind='cubic')
        states_nearest.append(state_nearest)
        states_interpolated_linear.append(state_interpolated_linear)
        states_interpolated_cubic.append(state_interpolated_cubic)
    return states_nearest, states_interpolated_linear, states_interpolated_cubic
