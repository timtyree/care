import numpy as np
from ._utils_find_contours import *


def plot_contours_pbc(contours, ax, linewidth=2, min_num_vertices=1, alpha=1., linestyle = '-'):
    for contour in contours:
        if len(contour)>=min_num_vertices:
            contour_lst = split_contour_into_contiguous(contour)
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
        contour_lst = split_contour_into_contiguous(contour)
        contour_lst_out = []
        #the first segment of this pbc contour
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

def _bring_vertices_together(vmove, vtarget, width, height):
    '''bring 2D vertex vmove as close as possible to 2D vertex 
    vtarget by increasing xy coordinates by width or height to decrease the distance from vmove to vtarget.'''
    vw = np.array([width,0.])
    vh = np.array([0.,height])
    dist = np.linalg.norm(vmove-vtarget)
    vtest = vmove + vw
    dist_test = np.linalg.norm(vtest-vtarget)
    if dist_test < dist:
        vmove = vtest.copy()
        dist = dist_test.copy()
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
        dist = dist_test.copy()
    else:
        vtest = vmove - vh
        dist_test = np.linalg.norm(vtest-vtarget)
        if dist_test < dist:
            vmove = vtest.copy()
            #dist = np.linalg.norm(vmove-vtarget)
    return vmove

def split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=2):
    '''returns a list of contiguous contours with one contour vertex added from across the computational domain, mapped onto the local coordinates. 
    jump_threshold = the max number of pixels two points may be separated by to be considered contiguous.
    size_threshold = the min number of vertices in a pbc contour
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

        #augment contours for each jump that occurred
        N = len(contour_lst)
        #for each jump, n
        for n in range(N): 
            ct_before       = contour_lst[n] #before jump
            if n<N-1: #this checks for a jump at the end/beginning of the pbc contour.  This shouldn't be used when the final vertex equals the initial vertex.
                ct_after       = contour_lst[n+1] #after jump
            else:
                ct_after       = contour_lst[0]
            vend     = ct_before[-1]
            vstart   = ct_after[0]
            #map the start to the end and the end to the start
            vmove=vstart
            vtarget=vend
            vmapped_end = _bring_vertices_together(vmove, vtarget, width=width, height=height)
            vmove=vend
            vtarget=vstart
            vmapped_start = _bring_vertices_together(vmove, vtarget, width=width, height=height)
            #add the new end to the end if it hasn't already
            if ~(vmapped_end==vend).all():
                contour_lst[n] = np.vstack([ct_before,vmapped_end])
            #add the new start to the start if it hasn't already
            if ~(vmapped_start==vstart).all():
                if n<N-1: #this checks for a jump at the end/beginning of the pbc contour.  This shouldn't be used when the final vertex equals the initial vertex.
                    contour_lst[n+1] = np.vstack([vmapped_start, ct_after])
                else:
                    contour_lst[0] = np.vstack([vmapped_start, ct_after])
        return contour_lst

#deprecated
# def unwrap_tips(n_lst, x_lst, y_lst):
#     '''unwrap n_lst to s1_values and s2_values'''
#     s1_values = []
#     s2_values = []
#     for n,lst in enumerate(x_lst):
#         m = len(lst)
#         s1, s2 = n_lst[n]
#         s1_values.extend([s1 for j in range(m)])
#         s2_values.extend([s2 for j in range(m)])
#     x_values = flatten(x_lst)
#     y_values = flatten(y_lst)
#     return s1_values, s2_values, x_values, y_values
#deprecated
# def unwrap_and_reduce_tips(n_lst, x_lst, y_lst, width, height, pad=1, decimals=11):
#     '''returns s1_values, s2_values, x_values, y_values each as numpy arrays
#     unwrap_and_reduce_tips unwraps output of find_tips and then removes any tip duplicates if they are within distance pad of the boundary.
#     duplicate tips are identified by sharing an x or y coordinate that match to machine precision, 
#     i.e. are tips duplicated by periodic boundary conditions.  In the event of such a duplicate tip, 
#     the right/bottom tip is retained.
#     '''
#     #unwrap tips
#     s1_values, s2_values, x_values, y_values = unwrap_tips(n_lst, x_lst, y_lst)
#     return reduce_tips(s1_values, s2_values, x_values, y_values, width, height, pad=pad, decimals=decimals)


def reduce_tips(s1_values, s2_values, x_values, y_values, width, height, pad=1, decimals=11):
    #reduce redundant tips
    # identify any duplicate values in each coord
    x_values_rounded = np.around(x_values, decimals=decimals)
    y_values_rounded = np.around(y_values, decimals=decimals)
    boo_x = np.array([(list(x_values_rounded).count(x)-1)>0 for x in x_values_rounded])
    boo_y = np.array([(list(y_values_rounded).count(y)-1)>0 for y in y_values_rounded])

    s1_values=np.array(s1_values) 
    s2_values=np.array(s2_values)
    x_values =np.array(x_values)
    y_values =np.array(y_values)

    #step 1: determine if any such duplicate tips exist
    duplicates_exist = boo_x.any() or boo_y.any()
    if not duplicates_exist:
        return s1_values, s2_values, x_values, y_values

    #step 2: remove any tip duplicates in the x coordinates if they are within distance pad of the boundary
    if boo_x.any():
        #remove any tip duplicates in the x coordinates if they are within distance pad of the boundary
        ar, index, inverse, counts = np.unique(x_values_rounded, return_index=True, return_inverse=True, return_counts=True,)
        # #test that the inverse map reconstructs the original data
        # assert ( (ar[inverse]-x_values_rounded == 0.).all() ) 
        # #test that the forward map, index constructs the unique data
        # assert ( (ar - x_values_rounded[index] == 0.).all() ) 

        indices_to_be_appended = []
        for j in index[counts>1]: 
            #get the value that corresponds to this count>1 instance is x_values[j]
            #get the original indices that correspond to this count>1 instance
            boo = x_values_rounded[j]==x_values_rounded
            #indices_to_be_appended at the end are those that lie in the bulk but still have a matching coordinate. The parameter pad controls this.
            #the index of any duplicate values that should not be eliminated because they lie outside a number of pixels = pad of the boundary
            indices_to_be_appended.extend(list(np.argwhere(boo & ( y_values > pad ) &  ( y_values < height-pad )).flatten()))
            #index to replace the one given
            i = int(np.argwhere(boo & ( y_values >= height-pad )).flatten())
        #append indices_to_be_appended, if any
        index = list(index)
        index.extend(indices_to_be_appended)

        #recompute the unwrapped tips, now guaranteed to be unique in this coordinate
        x_values = x_values[index]
        y_values = y_values[index]
        s1_values = s1_values[index]
        s2_values = s2_values[index]

        y_values_rounded = np.around(y_values, decimals=decimals)
        boo_y = np.array([(list(y_values_rounded).count(y)-1)>0 for y in y_values_rounded])

    #step 3: remove any tip duplicates in the y coordinates if they are within distance pad of the boundary
    if boo_y.any():

        ar, index, inverse, counts = np.unique(y_values_rounded, return_index=True, return_inverse=True, return_counts=True,)
        # #test that the inverse map reconstructs the original data
        # assert ( (ar[inverse]-y_values_rounded == 0.).all() ) 
        # #test that the forward map, index constructs the unique data
        # assert ( (ar - y_values_rounded[index] == 0.).all() ) 

        indices_to_be_appended = []
        for j in index[counts>1]: 
            #get the value that corresponds to this count>1 instance is y_values[j]
            #get the original indices that correspond to this count>1 instance
            boo = y_values_rounded[j]==y_values_rounded

            #indices_to_be_appended at the end are those that lie in the bulk but still have a matching coordinate. The parameter pad controls this.
            #the index of any duplicate values that should not be eliminated because they lie outside a number of pixels = pad of the boundary
            indices_to_be_appended.extend(list(np.argwhere(boo & ( x_values > pad ) &  ( x_values < width-pad )).flatten()))
            #index to replace the one given
            i = int(np.argwhere(boo & ( x_values >= width-pad )).flatten())
        #append indices_to_be_appended, if any
        index = list(index)
        index.extend(indices_to_be_appended)

        #recompute the unwrapped tips, now guaranteed to be unique in this coordinate
        x_values = x_values[index]
        y_values = y_values[index]
        s1_values = s1_values[index]
        s2_values = s2_values[index]
    return s1_values, s2_values, x_values, y_values



# def split_and_augment_contour_into_contiguous_segments(contour, width, height, threshold=2):
#     '''returns a list of contiguous contours with one contour vertex added from across the computational domain, mapped onto the local coordinates. 
#     jump_threshold = the max number of pixels two points may be separated by to be considered contiguous.
#     size_threshold = the min number of vertices in a pbc contour
#     split the contour into a contour_lst of contiguous contours.
#     the last contour may have length 1 or 0.  simply ignore those later.'''
    
#     mask = get_contiguous_mask(contour, threshold=threshold)
#     if mask.all():
#         #the contour is contiguous, no augmentation necessary.  Return the input contour as a list of one contour segment
#         return [contour]
#     else:
#         #the contour is not contiguous, jumps occured accross boundary
#         indices = np.nonzero(~mask)[0]+1
#         contour_lst = np.split(contour,indices)

#         #augment contours for each jump that occurred
#         N = len(contour_lst)
#         #for each jump, n
#         for n in range(N): 
#             ct_before       = contour_lst[n] #before jump
#             if n<N-1: #this checks for a jump at the end/beginning of the pbc contour.  This shouldn't be used when the final vertex equals the initial vertex.
#                 ct_after       = contour_lst[n+1] #after jump
#             else:
#                 ct_after       = contour_lst[0]
#             vend     = ct_before[-1]
#             vstart   = ct_after[0]
#             if vend.sum()>vstart.sum():
#                 #map vstart right and/or down to meet vend, appending to ct_before
#                 vmove=vstart
#                 vtarget=vend
#                 vmapped = _bring_vertices_together(vmove, vtarget, width=width, height=height)
#                 #append the mapped vstart to vend
#                 #informal test/check: print( f"Check {vmove} correctly moved to {vmapped} to meet {vtarget}.")
#                 #if vmapped is not exactly on top of vtarget
#                 if ~(vmapped==vtarget).all(): #this check makes the results robust to repeated application or inputs that are already augmented
#                     #then augment that contour with the mapped vertex
#                     contour_lst[n] = np.vstack([ct_before,vmapped])
#             else:
#                 #map vend right and/or down to meet vstart, prepending to ct_after
#                 vmove=vend
#                 vtarget=vstart
#                 vmapped = _bring_vertices_together(vmove, vtarget, width=width, height=height)
#                 #append the mapped vstart to vend
#                 #if vmapped is not exactly on top of vtarget
#                 if ~(vmapped==vtarget).all(): #this check makes the results robust to repeated application or inputs that are already augmented
#                     #informal test/check: print( f"Check {vmove} correctly moved to {vmapped} to meet {vtarget}.")
#                     #then augment that contour with the mapped vertex
#                     if n<N-1: #this checks for a jump at the end/beginning of the pbc contour.  This shouldn't be used when the final vertex equals the initial vertex.
#                         contour_lst[n+1] = np.vstack([vmapped, ct_after])
#                     else:
#                         contour_lst[0] = np.vstack([vmapped, ct_after])
#         return contour_lst

# def _bring_vertices_together(vmove, vtarget, width, height):
#     '''bring 2D vertex vmove as close as possible to 2D vertex 
#     vtarget by increasing xy coordinates by width or height to decrease the distance from vmove to vtarget.'''
#     vw = np.array([width,0.])
#     vh = np.array([0.,height])
#     dist = np.linalg.norm(vmove-vtarget)
#     vtest = vmove + vw
#     dist_test = np.linalg.norm(vtest-vtarget)
#     if dist_test < dist:
#         vmove = vtest.copy()
#         dist = dist_test.copy()
#     #     else:
#     #         vtest = vmove - vw
#     #         dist_test = np.linalg.norm(vtest-vtarget)
#     #         if dist_test < dist:
#     #             vmove = vtest.copy()
#     #             dist = np.linalg.norm(vmove-vtarget)
#     vtest = vmove + vh
#     dist_test = np.linalg.norm(vtest-vtarget)
#     if dist_test < dist:
#         vmove = vtest.copy()
#     #         dist = dist_test.copy()
#     #     else:
#     #         vtest = vmove - vh
#     #         dist_test = np.linalg.norm(vtest-vtarget)
#     #         if dist_test < dist:
#     #             vmove = vtest.copy()
#     #             #dist = np.linalg.norm(vmove-vtarget)
#     return vmove























