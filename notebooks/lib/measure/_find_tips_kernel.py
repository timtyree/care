import numpy as np
from ._find_tips_pbc_cy import lookup_segments
from numba import njit
#test based dev of needed functions
# @njit
def compute_theta(u,v):
    return np.abs(np.pi/2-np.arccos(np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))))

def _get_box(u,v):
    minx = np.min((np.min(u[0,:]),np.min(v[0,:])))
    maxx = np.max((np.max(u[0,:]),np.max(v[0,:])))
    miny = np.min((np.min(u[1,:]),np.min(v[1,:])))
    maxy = np.max((np.max(u[1,:]),np.max(v[1,:])))
    return minx, maxx, miny, maxy

def is_in_box(x,y,minx, maxx, miny, maxy):
    return ((x>=minx) & (maxx>=x)) & ((y>=miny) & (maxy>=y))

def intersection_2d_implicit(x1,y1,c1,x2,y2,c2):
    return _intersect_two_lines(x1,y1,c1,x2,y2,c2)
def _intersect_two_lines(a1,b1,c1,a2,b2,c2):
    '''asserts a*x + b*y + c = 0 for <-> collocate both lines.  note this is an involution.'''
    x = (b1*c2 - b2*c1)/(a1*b2 - a2*b1)
    y = (a2*c1 - a1*c2)/(a1*b2 - a2*b1)
    return x,y

def find_tips_for_linear_segment_pairs(segments1, segments2, r0,c0, theta_threshold=0.):
    '''lst_x,lst_y,lst_theta, lst_grad_ux,lst_grad_uy, lst_grad_vx,lst_grad_vy = find_tips_for_linear_segment_pairs(segments1, segments2)'''
    lst_x = []
    lst_y = []
    lst_grad_ux = []
    lst_grad_uy = []
    lst_grad_vx = []
    lst_grad_vy = []
    lst_theta = []
    #for each segment pair,
    for segment1 in segments1:
        for segment2 in segments2:
            u = np.array(segment1)
            v = np.array(segment2)
            # theta = -9999.#compute_theta(du,dv)
            theta = compute_theta(u[1]-u[0],v[1]-v[0])

            if np.abs(theta)<theta_threshold:
                continue


            #compute line for segment u
            x1=u[0][1];y1=u[0][0];x2=u[1][1];y2=u[1][0];
            c1 = - x1 - y1 #wlog
            a1,b1 = _intersect_two_lines(x1,y1,c1,x2,y2,c1)
            grad_ux = a1
            grad_uy = b1

            #compute line for segment v
            x1=v[0][1];y1=v[0][0];x2=v[1][1];y2= v[1][0];
            c2 = - x1 - y1 #wlog
            a2,b2 = _intersect_two_lines(x1,y1,c2,x2,y2,c2)
            grad_vx = a2
            grad_vy = b2

            #compute point of intersect
            x,y = _intersect_two_lines(a1,b1,c1,a2,b2,c2)

            # # #determing if the tip is in the right range
            c0 = int(np.floor(np.min(np.hstack((u[:,0],v[:,0])))))
            r0 = int(np.floor(np.min(np.hstack((u[:,1],v[:,1])))))
            minx,maxx,miny,maxy = r0,r0+1,c0,c0+1
            isin = is_in_box(x,y,minx, maxx, miny, maxy)

            #determing if the tip is in the right range
            # minx ,maxx, miny, maxy = _get_box(u,v)
            # isin = is_in_box(x,y,minx, maxx, miny, maxy)
            if isin:
                #record all of the data
                lst_x.append(x)
                lst_y.append(y)
                lst_grad_ux.append(grad_ux)
                lst_grad_uy.append(grad_uy)
                lst_grad_vx.append(grad_vx)
                lst_grad_vy.append(grad_vy)
                lst_theta.append(theta)
    return lst_x,lst_y,lst_theta, lst_grad_ux,lst_grad_uy, lst_grad_vx,lst_grad_vy

def find_intersections(array1,array2,level1,level2,theta_threshold = 0.):
    '''iterates over array1 (concurrently over array2) and find any line segments corresponding to the isolines level1 or level2.
    Considers only intersection points that intersect within the present pixel.
    Considers only intersection points that intersect at an angle of at least theta_threshold (radians).
    '''
    height,width = array1.shape
    lst_values_x = []
    lst_values_y = []
    lst_values_theta = []
    lst_values_grad_ux = []
    lst_values_grad_uy = []
    lst_values_grad_vx = []
    lst_values_grad_vy = []
    for c0 in range(height):
        for r0 in range(width):
            c1 = c0+1
            r1 = r0+1

            r1loc = r1
            if r1loc >= height:
                    r1loc = 0
            c1loc = c1
            if c1loc >= height:
                    c1loc = 0

            #find any segments for array1
            #if none are found, continue
            ul1 = array1[r0, c0]
            ur1 = array1[r0, c1loc]
            ll1 = array1[r1loc, c0]
            lr1 = array1[r1loc, c1loc]

            square_case1 = 0
            if (ul1 > level1): square_case1 += 1
            if (ur1 > level1): square_case1 += 2
            if (ll1 > level1): square_case1 += 4
            if (lr1 > level1): square_case1 += 8

            if square_case1 in [0, 15]:
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue    

            #find any segments for array2
            ul2 = array2[r0, c0]
            ur2 = array2[r0, c1loc]
            ll2 = array2[r1loc, c0]
            lr2 = array2[r1loc, c1loc]

            square_case2 = 0
            if (ul2 > level2): square_case2 += 1
            if (ur2 > level2): square_case2 += 2
            if (ll2 > level2): square_case2 += 4
            if (lr2 > level2): square_case2 += 8

            if square_case2 in [0, 15]:
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue

            segments1 = lookup_segments(ul1,ll1,ur1,lr1,r0,r1,c0,c1,level1,square_case1)
            segments2 = lookup_segments(ul2,ll2,ur2,lr2,r0,r1,c0,c1,level2,square_case2)

            lst_x,lst_y,lst_theta, lst_grad_ux, lst_grad_uy, lst_grad_vx, lst_grad_vy = find_tips_for_linear_segment_pairs(segments1, segments2, r0, c0)

            lst_values_x.extend(lst_x)
            lst_values_y.extend(lst_y)
            lst_values_theta.extend(lst_theta)
            lst_values_grad_ux.extend(lst_grad_ux)
            lst_values_grad_uy.extend(lst_grad_uy)
            lst_values_grad_vx.extend(lst_grad_vx)
            lst_values_grad_vy.extend(lst_grad_vy)
    return lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy


