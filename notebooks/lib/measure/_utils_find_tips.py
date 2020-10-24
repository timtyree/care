import numpy as np
from ._utils_find_contours import *


def plot_contours_pbc(contours, ax, linewidth=2, min_num_vertices=1, alpha=1., linestyle = '-'):
    for contour in contours:
        contour_lst = _split_contour_into_contiguous(contour)
        #plot the first segment of this pbc contour
        ct = contour_lst[0]
        if len(ct)>min_num_vertices:
            p = ax.plot(ct[:, 1], ct[:, 0], linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        #if there are more segments, plot them as well.
        if len(contour_lst)>1:
            color = p[0].get_color()
            for ct in contour_lst[1:]:
                if len(ct)>min_num_vertices:
                    ax.plot(ct[:, 1], ct[:, 0], linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)
def segment_and_filter_short_contours(contours, min_num_vertices=2):
    '''avoid using this function directly, as it destroys contour index information'''
    contours_lst_out = []
    for contour in contours:
        contour_lst = _split_contour_into_contiguous(contour)
        contour_lst_out = []
        #plot the first segment of this pbc contour
        ct = contour_lst[0]
        if len(ct)>=min_num_vertices:
            contour_lst_out.append(ct)
        #if there are more segments, plot them as well.
        if len(contour_lst)>1:
            for ct in contour_lst[1:]:
                if len(ct)>=min_num_vertices:
                    contour_lst_out.append(ct)
        contours_lst_out.append(contour_lst_out)
    return contours_lst_out

def _flatten(x_lst):
    xlist = []
    for xl in x_lst:
        xlist.extend(xl)
    return xlist

def flatten(x_lst):
    return _flatten(x_lst)

def split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=2):
    '''returns a list of contiguous contours with one contour vertex added from across the computational domain, mapped onto the local coordinates. 
    threshold = the max number of pixels two points may be separated by to be considered contiguous.
    split the contour into a contour_lst of contiguous contours.
    the last contour may have length 1 or 0.  simply ignore those later.'''
    mask = get_contiguous_mask(contour, threshold=threshold)
    if mask.all():
        #the contour is contiguous, no augmentation necessary.  Return the input contour as a list of one contour segment
        return [contour]
    else:
        #the contour is not contiguous, jumps occured accross boundary
        indices = np.nonzero(~mask)[0]+1
        contour_lst = np.split(contour,indices)
        
        #augment the end of each contiguous segment with the start of the following segment, if there is a following segment
        for n in range(len(contour_lst)-1):
            ct      = contour_lst[n]
            vtarget = ct[-1]
            vmove   = contour_lst[n+1][0]
            #map the next vertex onto the current vertex
            vmapped = _bring_vertices_together(vmove, vtarget, width, height)
            
            #print( f"Check {vmove} correctly moved to {vmapped} to meet {vtarget}.")
            #if vmapped is not exactly on top of vtarget
            if ~(vmapped==vtarget).all():
                #augment that contour with the mapped vertex
                contour_lst[n] = np.vstack([ct,vmapped])
        return contour_lst


def _bring_vertices_together(vmove, vtarget, width, height):
    '''bring 2D vertex vmove as close as possible to 2D vertex 
    vtarget by changing xy coordinates by width or height.'''
    vw = np.array([width,0.])
    vh = np.array([0.,height])
    dist = np.linalg.norm(vmove-vtarget)
    vtest = vmove + vw
    dist_test = np.linalg.norm(vtest-vtarget)
    if dist_test < dist:
        vmove = vtest.copy()
        dist = np.linalg.norm(vmove-vtarget)
    else:
        vtest = vmove - vw
        dist_test = np.linalg.norm(vtest-vtarget)
        if dist_test < dist:
            vmove = vtest.copy()
            dist = np.linalg.norm(vmove-vtarget)
    vtest = vmove + vh
    dist_test = np.linalg.norm(vtest-vtarget)
    if dist_test < dist:
        vmove = vtest.copy()
        dist = np.linalg.norm(vmove-vtarget)
    else:
        vtest = vmove - vh
        dist_test = np.linalg.norm(vtest-vtarget)
        if dist_test < dist:
            vmove = vtest.copy()
            #dist = np.linalg.norm(vmove-vtarget)
    return vmove























