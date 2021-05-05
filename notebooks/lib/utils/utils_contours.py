import numpy as np
from .dist_func import get_distance_L2_pbc
from numba import njit
from .projection_func import *
# Programmer: Tim Tyree
# Date: 4.29.2021

def get_comp_perimeter(width=200.,height=200.):
    '''returns project_point_2D
    Obeys periodic boundary conditions on a width-x-height square domain.

    returns the fraction of the length of segment where the projection of point onto segment lies.
    returns 0. when point=segment[0].
    returns a value in the interval [0,1) if this point lies approximately within this segment
    Example Usage:
    project_point_2D=get_project_point_2D(width=200,height=200)
    frac=project_point_2D(point,segment)
    '''
    distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)
    @njit
    def comp_perimeter(contour):
        Nseg=contour.shape[0]
        arclen=0.
        for i in range(-1,Nseg-1):
            arclen+=distance_L2_pbc(contour[i],contour[i+1])
        return arclen #arclength in pixels
    return comp_perimeter

#DONE: compute the node indices for all spiral tips
#DONE(later): integrate ^this into a function
def get_fix_node_id(width=200.,height=200.):
    project_point_2D=get_project_point_2D(width=width,height=height)
    def fix_node_id(contour,point,node_id):
        node_id_out=node_id
        segment=contour[node_id_out:node_id_out+2]
        frac=project_point_2D(point, segment)
        if not (frac>=0)&(frac<1):
            # print('closest node index is not valid')
            node_id_out=node_id-1
            segment=contour[node_id_out:node_id_out+2]
            frac=project_point_2D(point, segment)
            if not (frac>=0)&(frac<1):
                # print('prev is not valid')
                node_id_out=node_id+1
                segment=contour[node_id_out:node_id_out+2]
                frac=project_point_2D(point, segment)
                if not (frac>=0)&(frac<1):
                    # print('next is not valid')
                    node_id_out=node_id+2
                    segment=contour[node_id_out:node_id_out+2]
                    frac=project_point_2D(point, segment)
                    if not (frac>=0)&(frac<1):
                        # print('next next is not valid')
                        node_id_out=node_id-2
                        segment=contour[node_id_out:node_id_out+2]
                        frac=project_point_2D(point, segment)
                        if not (frac>=0)&(frac<1):
                            # print('prev prev prev is not valid')
                            node_id_out=node_id-3
                            segment=contour[node_id_out:node_id_out+2]
                            frac=project_point_2D(point, segment)
                            if not (frac>=0)&(frac<1):
                                # print('next next next is not valid')
                                node_id_out=node_id+3
                                segment=contour[node_id_out:node_id_out+2]
                                frac=project_point_2D(point, segment)
                                if not (frac>=0)&(frac<1):
                                    # print('prev prev is not valid')
                                    # print('no valid start found... returning input node_id...')
                                    # node_id_out=node_id
                                    print('.', end='\r')
                                    # print("Warning: no valid start found for input")
                                    # raise Exception(f'no valid start found for input, contour,point,node_id={(contour,point,node_id)}')
                                    node_id_out=node_id
        return node_id_out
    return fix_node_id

def get_segment_pbc(node_start,N_nodes,contour):
    na=(node_start)   % N_nodes
    nb=(node_start+2) % N_nodes
    if nb>na:
        segment=contour[na : nb]
        # assert(segment.shape==(2,2))
    else: # edge case segment
        Q=contour[-1]
        W=contour[0]
        segment=np.stack([Q,W])
        # assert(segment.shape==(2,2))
    return segment

def compare_spiralarm_size(j_lst, j_nxt_lst, archlen_size_lst):
    '''supposes pid_lst=list(range(ntips))
    Example Usage:
    greater_i_lst,lesser_i_lst=compare_spiralarm_size(j_lst, j_nxt_lst, archlen_size_lst)
    '''
    ntips=len(j_lst)
    #iterate over archlen observations
    pid_lst=list(range(ntips))
    pid_to_i     ={}
    pid_to_i_nxt ={}
    for i in pid_lst:#range(len(j_lst)):
        pid_to_i.update({pid_lst[j_lst[i]]:i})         # map from i^th observation start to spiral tip index
        pid_to_i_nxt.update({pid_lst[j_nxt_lst[i]]:i}) # map from i^th observation end to spiral tip index
    # print(pid_to_i)
    # print(pid_to_i_nxt)

    #iterate over particles
    greater_i_lst=[]
    lesser_i_lst=[]
    for pid in pid_lst:
        # try:
        size_right = archlen_size_lst[pid_to_i[pid]]
        size_left  = archlen_size_lst[pid_to_i_nxt[pid]]
        # except Exception as e:
        #     print(pid,archlen_size_lst,pid_to_i,pid_to_i_nxt)
        #     raise(e)
        if size_right>size_left:
            greater_i=pid_to_i[pid]
            lesser_i =pid_to_i_nxt[pid]
            #         greater_archlen_values=archlen_values_lst[greater_i]
            #         lesser_archlen_values=archlen_values_lst[lesser_i]
        else:
            lesser_i=pid_to_i[pid]
            greater_i =pid_to_i_nxt[pid]
        greater_i_lst.append(greater_i)
        lesser_i_lst.append(lesser_i)
    return greater_i_lst,lesser_i_lst



def compare_spiralarm_voltage(j_lst, j_nxt_lst, avgVoltage_lst):
    '''supposes pid_lst=list(range(ntips))
    Example Usage:
    greater_i_lst,lesser_i_lst=compare_spiralarm_size(j_lst, j_nxt_lst, archlen_size_lst)
    '''
    ntips=len(j_lst)
    #iterate over archlen observations
    pid_lst=list(range(ntips))
    pid_to_i     ={}
    pid_to_i_nxt ={}
    for i in pid_lst:#range(len(j_lst)):
        pid_to_i.update({pid_lst[j_lst[i]]:i})         # map from i^th observation start to spiral tip index
        pid_to_i_nxt.update({pid_lst[j_nxt_lst[i]]:i}) # map from i^th observation end to spiral tip index
    # print(pid_to_i)
    # print(pid_to_i_nxt)

    #iterate over particles
    greater_i_lst=[]
    lesser_i_lst=[]
    for pid in pid_lst:
        # try:
        size_right = avgVoltage_lst[pid_to_i[pid]]
        size_left  = avgVoltage_lst[pid_to_i_nxt[pid]]
        # except Exception as e:
        #     print(pid,archlen_size_lst,pid_to_i,pid_to_i_nxt)
        #     raise(e)
        if size_right>size_left:
            greater_i=pid_to_i[pid]
            lesser_i =pid_to_i_nxt[pid]
            #         greater_archlen_values=archlen_values_lst[greater_i]
            #         lesser_archlen_values=archlen_values_lst[lesser_i]
        else:
            lesser_i=pid_to_i[pid]
            greater_i =pid_to_i_nxt[pid]
        greater_i_lst.append(greater_i)
        lesser_i_lst.append(lesser_i)
    return greater_i_lst,lesser_i_lst
