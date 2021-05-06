import numpy as np
# from ._utils_find_contours import *
from .intersection import *
# from ._utils_find_tips import *
# from .. import *
from ._utils_find_tips import split_and_augment_contour_into_contiguous_segments

def find_tips(contours1, contours2):
    '''returns tips with indices of parent contours.
    contours1 and contours2 are a lists of numpy arrays.
    each such numpy array is an Nx2 array representing a contiguous contour (i.e. continuous in local xy coordinates).'''
    s1_list = []; s2_list = []; x_lst = []; y_lst = []
    for n1, contour1_lst in enumerate(contours1):
        for c1 in contour1_lst:
            for n2, contour2_lst in enumerate(contours2):
                for c2 in contour2_lst:
                    x1, y1 = (c1[:, 0], c1[:, 1])
                    x2, y2 = (c2[:, 0], c2[:, 1])
                    #intersection is not the cause of the slowdown
                    yl, xl = intersection(x1, y1, x2, y2)
                    for x,y in zip(xl,yl):
                        s1_list.append(n1)
                        s2_list.append(n2)
                        x_lst.append(x)
                        y_lst.append(y)
    #DONE: sort####compute node_id_values=?? using contour1_len_values and locally determined values in the sublist
    sorted_values=np.array(sorted(zip(x_lst,y_lst,s1_lst,s2_lst))))
    in_to_sorted_values=np.argsort(sorted_values)
    x_lst, y_lst, s1_lst, s2_lst=in_to_sorted_values
    return s1_list, s2_list, x_lst, y_lst

def find_tips_with_pbc_knots(contours1, contours2, s1in_lst, s2in_lst):
    '''returns tips with indices of parent contours.
    contours1 and contours2 are a lists of numpy arrays.
    each such numpy array is an Nx2 array representing a contiguous contour (i.e. continuous in local xy coordinates).'''
    s1_list = []; s2_list = []; x_lst = []; y_lst = []
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
    #DONE: sort####compute node_id_values=?? using contour1_len_values and locally determined values in the sublist
    sorted_values=np.array(sorted(zip(x_lst,y_lst,s1_lst,s2_lst))))
    in_to_sorted_values=np.argsort(sorted_values)
    x_lst, y_lst, s1_lst, s2_lst=in_to_sorted_values
    return s1_list, s2_list, x_lst, y_lst

def preprocess_contours(contours1,contours2,width, height,jump_threshold = 2,size_threshold = 6):
    #segment and augment contours of the first family
    contour1_lst_lst = []
    for contour in contours1:
        if len(contour) >= size_threshold:
            contour1_lst = split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=jump_threshold)
            contour1_lst_lst.append(contour1_lst)
    #segment and augment contours of the second family
    contour2_lst_lst = []
    for contour in contours2:
        if len(contour) >= size_threshold:
            contour2_lst = split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=jump_threshold)
            contour2_lst_lst.append(contour2_lst)
    return contour1_lst_lst, contour2_lst_lst

def preprocess_contours_and_enumerate(contours1,contours2,width, height,jump_threshold = 2,size_threshold = 6):
    '''Example Usage:
    contour1_lst_lst, contour2_lst_lst, s1_lst, s2_lst = preprocess_contours_and_enum(contours1,contours2,width, height,jump_threshold = 2,size_threshold = 6)
    '''
    #segment and augment contours of the first family
    contour1_lst_lst = [];s1_lst=[]
    for s,contour in enumerate(contours1):
        if len(contour) >= size_threshold:
            contour1_lst = split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=jump_threshold)
            contour1_lst_lst.append(contour1_lst)
            s1_lst.append(s)
    #segment and augment contours of the second family
    contour2_lst_lst = [];s2_lst=[]
    for s, contour in enumerate(contours2):
        if len(contour) >= size_threshold:
            contour2_lst = split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=jump_threshold)
            contour2_lst_lst.append(contour2_lst)
            s2_lst.append(s)
    return contour1_lst_lst, contour2_lst_lst, s1_lst, s2_lst

def contours_to_simple_tips_pbc(contours1,contours2,width, height,jump_threshold = 10,size_threshold = 4):
    '''Note: topological values may be incorrect...

    Find the intersection points of two families of contour lines.
    contours_to_simple_tips_pbc returns spiral tips as a tuple of lists, n_lst, x_lst, y_lst.
    n_lst = list of tuples with the first tuple value is the index of the parent contour1 and the second tuple value is the index of the parent contour2.
    x_lst = list of lists of intersection x coordinates cooresponding to the tuples in n_lst.
    y_lst = list of lists of intersection y coordinates cooresponding to the tuples in n_lst.
    width, height = the number of rows?,columns? of the input image
    jump_threshold = max distance in pixels to not be considered a jump
    size_threshold = minimum number of vertices in a whole input contour to be considered

    pixels are taken to be centered at integer xy coordinates.
    (nota bene for my original use case: a lot of 5 vertex contours were observed in contours2,
    where contours1 and contour2 are results of lewiner marching squares (that used explicit pbc).  )
    '''
    contour1_lst_lst, contour2_lst_lst, s1in_lst, s2in_lst = preprocess_contours_and_enumerate(contours1, contours2, width, height, jump_threshold = jump_threshold, size_threshold = size_threshold)

    #detect tips as above and return tip number as above
    #DONE: at the end of find_tips_with_pbc_knots, return the net node indices of the contour1_lst_lst, which may be used later with contour1_len_lst
    contour1_len_values=np.array([len(x) for x in contour1_lst_lst])
    s1_lst, s2_lst, x_lst, y_lst = find_tips_with_pbc_knots(contour1_lst_lst, contour2_lst_lst, s1in_lst, s2in_lst)
    #TODO: compute node_id_values=?? using contour1_len_values and locally determined values in the sublist

    return s1_lst, s2_lst, x_lst, y_lst

# @njit
def find_tips_wrapped(contours1, contours2):
    '''(deprecated)
    returns tips with indices of parent contours.
    contours1 and contours2 are a lists of numpy arrays.
    each such numpy array is an Nx2 array representing a contiguous contour (i.e. continuous in local xy coordinates).'''
    List()
    #     n_list = []; x_lst = []; y_lst = []
    for n1, contour1_lst in enumerate(contours1):
        #contour1_lst = split_contour_into_contiguous(contour1)
        for c1 in contour1_lst:
            for n2, contour2_lst in enumerate(contours2):
                #contour2_lst = split_contour_into_contiguous(contour2)
                for c2 in contour2_lst:
                    x1, y1 = (c1[:, 0], c1[:, 1])
                    x2, y2 = (c2[:, 0], c2[:, 1])
                    # try:
                    #     y, x = intersection(x1, y1, x2, y2)
                    # except Exception as e:
                    #     print(e)
                    #     print(f"x1 = {x1}")
                    #     print(f"y1 = {y1}")
                    #     print(f"x2 = {x2}")
                    #     print(f"y2 = {y2}")
                    y, x = intersection(x1, y1, x2, y2)
                    if len(x)>0:
                        s = (n1,n2)
                        x = list(x)
                        y = list(y)
                        n_list.append(s)
                        x_lst.append(x)
                        y_lst.append(y)
    #DONE: sort####compute node_id_values=?? using contour1_len_values and locally determined values in the sublist
    sorted_values=np.array(sorted(zip(x_lst,y_lst,s1_lst,s2_lst))))
    in_to_sorted_values=np.argsort(sorted_values)
    x_lst, y_lst, s1_lst, s2_lst=in_to_sorted_values
    return n_list, x_lst, y_lst


####################
# Example usage from raw texture
####################
# #compute as discrete flow map dtexture_dt
# zero_txt = txt.copy()*0.

# #calculate
# dtexture_dt = zero_txt.copy()
# get_time_step(txt, dtexture_dt)

# #compute the images to find isosurfaces of
# img    = txt[...,0]
# dimgdt = dtexture_dt[...,0]

# #compute both families of contours
# contours1 = find_contours(img, level = 0.5)
# contours2 = find_contours(dimgdt, level = 0.0)

# #find_tips
# n_list, x_lst, y_lst = find_tips(contours1, contours2)

# num_tips = count_tips(x_lst)
# x_values = np.array(_flatten(x_lst))
# y_values = np.array(_flatten(y_lst))
# # y_values = height-1.*np.array(_flatten(y_lst))
