import pandas as pd, numpy as np
from .. import *
from ..utils.dist_func import get_distance_L2_pbc
from ..measure.relative_phases import compute_phi_values
from ..measure.full_color_contours import get_comp_dict_topo_full_color
from ..measure._find_tips_kernel_cy import find_intersections
def comp_dict_simp(img,img_prev,V_threshold):
    '''
    Example Usage:
    dict_simp=comp_dict_simp(img,img_prev,V_threshold)
    '''
    lst_values_x,lst_values_y,_, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy=find_intersections(array1=img,array2=img_prev,level1=V_threshold,level2=V_threshold)
    dict_simp={
        'x':lst_values_x,
        'y':lst_values_y,
        'grad_ux':lst_values_grad_ux,
        'grad_uy':lst_values_grad_uy,
        'grad_vx':lst_values_grad_vx,
        'grad_vy':lst_values_grad_vy
    }
    return dict_simp


def get_sort_particles_indices(width,height):
    dist_L2_pbc=get_distance_L2_pbc(width=width,height=height)
    def sort_particles_indices(dict_out, dict_in, search_range=40.):
        '''sorts from dict_topo indicies to dict_simp indicies.  returns dict
        DONE: make this map from dict_simp to dict_topo
        TODO: tune search_range using an estimate of max displacement between observations

        Example Usage:
        in_to_out=sort_particles_indices(dict_topo, dict_simp, search_range=40.)
        '''
        dict_topo=dict_out
        dict_simp=dict_in
        #TODO(later): switch the names of dict_simp/topo in the rest of this function, then swap ^that.
        #   As it is, the instantaneous position is being recorded
        pid_out_lst=list(range(len(list(dict_simp['x']))))
        pid_in_lst =dict_topo['pid']
        x_lst=dict_topo['x'];y_lst=dict_topo['y'];
        x2_lst=dict_simp['x'];y2_lst=dict_simp['y'];
        # make dict that maps dict_tips['pid'] to nearest list(self.keys()) that has tmax as its time
        in_to_out={}
        #for each point[pid_in]
        for pid_in in pid_in_lst:
            point=np.array((x_lst[pid_in],y_lst[pid_in]))
            mindist=9e9
            pid_out_closest=-9999
            #find the closest position[pid_out]
            for pid_out in pid_out_lst:
                point_out=np.array((x2_lst[pid_out],y2_lst[pid_out]))
                dist=dist_L2_pbc(point,point_out)
                if dist<mindist:
                    pid_out_closest=pid_out
                    mindist=dist
            if mindist<search_range:
                pid_out_lst.remove(pid_out_closest)
                #and update in_to_out with that pid_out
                in_to_out.update({pid_in:pid_out_closest})
        return in_to_out
    return sort_particles_indices

# def get_map_simp_to_topo(width,height):
#     sort_particles_indices=get_sort_particles_indices(width,height)
#     dist_L2_pbc=get_distance_L2_pbc(width=width,height=height)
#     def map_simp_to_topo(dict_simp,dict_topo):
#         x_valid_values=np.array(dict_simp['x'])
#         y_valid_values=np.array(dict_simp['y'])
#         x_basket_values=np.array(dict_topo['x'])
#         y_basket_values=np.array(dict_topo['y'])
#         dict_in_to_out={}
#         dict_in_to_dist={}
#         queue=list(enumerate(zip(x_basket_values,y_basket_values)))
#         for pid,point1 in enumerate(zip(x_valid_values,y_valid_values)):
#             #find the nearest tip remaining in x_basket_values
#             while len(queue)>0:
#                 mindist=9999.
# # pop when there's a match                j,point2 = queue.pop()
#                 dist=dist_L2_pbc(np.array(point1),np.array(point2))
#                 if dist<mindist:
#                     mindist=dist
#                     minpid_neighbor=j
#             if mindist<9999.:
#                 #record the champion neighbor
#                 dict_in_to_out.update({pid:minpid_neighbor})
#                 dict_in_to_dist.update({pid:mindist})
#                 j,point2 = queue.pop()

#         return dict_in_to_out,dict_in_to_dist
#     return map_simp_to_topo

def zip_tips(dict_simp,dict_topo,dict_in_to_out):
    primitive = (int, str, bool, float)
    def is_primitive(thing):
        return isinstance(thing, primitive)

    dict_tips={};pid_lst=[]
    pid_in_lst=list(dict_in_to_out.keys())
    #TODO: make from dict_simp to dict_topo
    #TODO: ensure the final pid being recorded is from dict_simp
    #for each item referenced from dict_in_to_out, update first with the full color results
    for pid in pid_in_lst:
        pid_out=dict_in_to_out[pid]
        for key in dict_topo.keys():
            value=dict_topo[key]
            if is_primitive(value):
                dict_tips[key]=value
            # elif :#heretim?
            elif set(dict_tips.keys()).issuperset({key}):
                dict_tips[key].append(value[pid])
            else:
                dict_tips[key]=[value[pid]]
            pid_lst.append(pid)

    #if key is not in dict_tips, update it with all pid values
    new_keys=list(set(dict_simp.keys()).difference(set(dict_topo.keys())))
    for key in new_keys:
        for pid in pid_in_lst:
            pid_out=dict_in_to_out[pid]
            try:
                dict_tips[key].append(dict_simp[key][pid_out])
            except (TypeError, KeyError) as e:
                dict_tips[key]=dict_simp[key]
            except KeyError as e:
                #initiate the field if it does not already exist
                dict_tips[key]=[dict_simp[key][pid_out]]
            except TypeError as e:
                pass
            pid_lst.append(pid)
    dict_tips['pid']=pid_lst
    return dict_tips

def get_comp_tips(width,height,V_threshold,search_range=40.):
    '''
    Example Usage:
    comp_tips=get_comp_tips(width,height,V_threshold)
    dict_tips=comp_tips(img,img_prev,dimgdt, t, txt)
    '''
    #TODO(later): compute level sets of all 2D arrays... and save the runtime of computing img's level set twice...
    sort_particles_indices=get_sort_particles_indices(width,height)
    comp_dict_topo_full_color=get_comp_dict_topo_full_color(width=width,height=height,level1=V_threshold,level2=0.)
    def comp_tips(img,img_prev,dimgdt, t, txt):
        #compute intersection points of level sets of img and img_prev
        dict_simp=comp_dict_simp(img,img_prev,V_threshold)
        #compute intersection points of level sets of img and dimgdt
        dict_topo=comp_dict_topo_full_color(img, dimgdt, t, txt)
        #for each tip from (img and img_prev), find the nearest tip from (img and dimgdt)
        #         dict_in_to_out,dict_in_to_dist=map_simp_to_topo(dict_simp,dict_topo)
        in_to_out=sort_particles_indices(dict_topo, dict_simp, search_range=search_range)
        #return the dict of all ^those tips
        dict_tips=zip_tips(dict_simp,dict_topo,in_to_out)
        #TODO: handle when lesser_pid and/or greater_pid is not included in the list
        #TODO: test reindex dict_tips works
        ntips=len(dict_tips['pid'])
        pid_map={}
        for n in range(ntips):
            pid=dict_tips['pid'][n]
            pid_map[pid]=n
            dict_tips['pid'][n]=pid_map[pid]
        for m,pid in enumerate(dict_tips['lesser_pid']):
            try:
                n=pid_map[pid]
            except KeyError as e:
                n=-1
            dict_tips['lesser_pid'][m]=n
        for m,pid in enumerate(dict_tips['greater_pid']):
            try:
                n=pid_map[pid]
            except KeyError as e:
                n=-1
            dict_tips['greater_pid'][m]=n        # for pid in dict_tips['greater_pid']:
        #     n=pid_map[pid]
        #     dict_tips['greater_pid'][n]=pid_map[dict_tips['greater_pid'][n]]
        #compute relative phase information along the lesser front
        try:
            phi_lst=compute_phi_values(dict_tips)
        except IndexError as e:
            print((dict_tips['pid'],dict_tips['x'],dict_tips['y']))
            raise(e)
        dict_tips['phi']=phi_lst

        return dict_tips
    return comp_tips
