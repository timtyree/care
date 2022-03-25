from .. import *
import numpy as np
from ..controller.controller_LR import get_one_step_explicit_synchronous_splitting
from ..measure.full_color_contours import get_comp_dict_topo_full_color
from ..utils.dist_func import get_distance_L2_pbc

def init_methods(width,height,ds,dt,nb_dir,V_threshold=-50,jump_threshold=40,diffCoef=0.0005,Cm=1.0,**kwargs):
    '''
    Example Usage:
    one_step,comp_distance,comp_dict_tips=init_methods(width,height,ds,dt,V_threshold=-50,jump_threshold=40)
    '''
    #TODO: pass kwargs to the get_one_step in use
    # get_one_step
    __, arr39, one_step = get_one_step_explicit_synchronous_splitting(
        nb_dir,dt,width,height,ds,diffCoef=diffCoef,Cm=Cm,**kwargs)

    #the heavyweight spiral tip measures
    comp_dict_tips=get_comp_dict_topo_full_color(width=width,height=height,level1=V_threshold,level2=0,
                                                 jump_threshold=jump_threshold,ds=ds)
    # comp_dict_tips=get_compute_all_spiral_tips(width, height, mode='simp')
    comp_distance=get_distance_L2_pbc(width,height)
    return one_step,comp_distance,comp_dict_tips

def zoom_system(txt,pdict):
    txt=zoomin_txt(txt)
    pdict=pdict.zoom_to_double()
    return txt,pdict

def zoom_in(txt,pdict,dt,level1,level2,tfactor=0.1,**kwargs):
    '''
    Example Usage:
    txt_prev, pdict, comp_tips, one_step=zoom_in(txt_prev,pdict,dt,level1,level2,tfactor=0.1)
    '''
    # zooming in as needed, until a certain absolute tolerance is found in a zero minimum distance between tips is reached.
    dt=tfactor*dt
    width,height=txt.shape[:2]
    comp_tips=get_comp_dict_topo_full_color(width=width,height=height,level1=level1,level2=level2)
    # get_one_step at this dt (Luo-Rudy model)
    __, arr39, one_step = get_one_step_explicit_synchronous_splitting(
        nb_dir,dt=dt,width=width,height=height,ds=ds,diffCoef=0.0005,Cm=1.0)

    txt,pdict=zoom_system(txt,pdict)
    return txt, pdict, dt, comp_tips, one_step

def compute_last_sigma_max(pdict,ds=5.):
    pid_lst=pdict.get_alive_particles()
    last_particle=pdict[pid_lst[0]]
    scale=ds/last_particle.width #cm/pixel
    sigma_max=scale*last_particle['lesser_arclen'][-1]
    return sigma_max

def find_stopping_point(dt,pdict,txt_prev,t_prev,save_every_n_frames=1,V_threshold=-50.,pid_pair=(0,1),ds=5.,
    one_step=None,comp_dict_topo_full_color=None,**kwargs):
    ''' of this pid_pair.  Supposes pid_pair denotes a birth
    Example Usage:
    txt_prev,t_prev,min_sigma_max=find_stopping_point(dt,pdict,txt_prev,t_prev,save_every_n_frames=1,V_threshold=-50.,pid_pair=(0,1),ds=5.)
    '''
    width,height=txt_prev.shape[:2]
    # get_one_step at this dt,ds
    # comp_dict_topo_simple=get_comp_dict_topo_simple(width=width,height=height)
    if one_step is None:
        __, __, one_step = get_one_step_explicit_synchronous_splitting(nb_dir,
                                                                       dt=dt,width=width,height=height,
                                                                       ds=5.,diffCoef=0.001,Cm=1.0)
    if comp_dict_topo_full_color is None:
        comp_dict_topo_full_color=get_comp_dict_topo_full_color(width=width,height=height,level1=V_threshold,level2=0)

    #reset at last observed time of spiral tip
    inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt_prev.copy())
    t=t_prev
    #compute nonlocal spiral tip observations
    img=inVc[...,0];dimgdt=dVcdt[...,0]
    # dict_topo=comp_dict_topo_simple(img,dimgdt,t)
    dict_topo=comp_dict_topo_full_color(img,dimgdt,t_prev,txt_prev)
    ntips=len(dict_topo['x'])

    pid_lst_living=pdict.get_alive_particles()
    tracked_pair_exists=set(pid_pair).issubset(set(pid_lst_living))

    # track tips between frames using ParticleSet while n_tips>0
    # while ntips>0:
    while tracked_pair_exists:
        # copy current texture
        txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        txt_prev=txt.copy()
        t_prev=t

        # one_step forward
        for step in range(save_every_n_frames):
            one_step(inVc, outVc, inmhjdfx, outmhjdfx, dVcdt)
            t+=dt

        #compute nonlocal spiral tip observations
        img=inVc[...,0];dimgdt=dVcdt[...,0]
        # dict_topo=comp_dict_topo_simple(img,dimgdt,t)
        dict_topo=comp_dict_topo_full_color(img,dimgdt,t,txt)

        #compute whether target particles are present
        in_to_out=pdict.sort_particles_indices(dict_topo)#, search_range=40.)
        pid_lst_living=list(in_to_out.values())
        tracked_pair_exists=set(pid_pair).issubset(set(pid_lst_living))
        ntips=len(dict_topo['x'])
        # if ntips>0:
        if tracked_pair_exists:
            #record nonlocal spiral tip observations
            pdict.merge_dict(dict_topo)
        else:
            print(f'death event found for pid_pair={pid_pair} at time t={t}, where dt={dt} and L={img.shape[0]}...')

    min_sigma_max=compute_last_sigma_max(pdict,ds)
    print(f"\t min_sigma_max={min_sigma_max} cm")
    return txt_prev,t_prev,min_sigma_max

def find_starting_point(pdict,pid_pair):
    '''find_starting_point returns xy position of the first particle in pid_pair.
    suppose the pid_pair are birth partners.
    follow pdict back to the birth of this pid_pair.
    '''
    particle=pdict[pid_pair[0]]
    xy0=np.array((particle['x'][0],particle['y'][0]))
    return xy0

def check_atol(pdict,pid_pair,atol,index=-1,ds=5.):
    particle1=pdict[pid_pair[0]]
    particle2=pdict[pid_pair[1]]
    point=np.array((particle1['x'][index],particle1['y'][index]))
    scale=ds/particle1.width
    dist=particle2.dist_to(point)*scale
    boo=dist<atol
    return boo

# def recursive_zoom_to_death(pid_pair_lst,txt_in,dt,V_threshold=-50.,atol=1e-2,
#                             one_steps_per_frame=100,ds=5.,
#                             **kwargs):
#                             ''''''
#     retval_lst=[]
#     #### brute force method is deprecated
#     # for pid_pair in pid_pair_lst:
#     #     txt_prev=txt_in.copy()
#     #     # one_step,comp_distance,comp_dict_tips=init_methods(txt_in,ds,dt,V_threshold=-50,jump_threshold=40)
#     #
#     #     txt_prev,t_prev,min_sigma_max=find_stopping_point(dt,pdict,txt_prev,t_prev,
#     #                                                       save_every_n_frames=one_steps_per_frame,
#     #                                                       V_threshold=V_threshold,
#     #                                                       pid_pair=pid_pair,ds=ds)
#     #     t_death=t_prev+dt
#     #     system=list(zip(t_death,pid_pair,txt_prev))
#     #     boo=check_atol(pdict,pid_pair,atol,index=-1,ds=ds)
#     #     while boo:
#     #         system=zoom_system(system)
#     #         t_death,pid_pair,txt_prev=system
#     #         dt/=10.
#     #         #TODO: compute each of pdict,txt_prev, t_prev=t_death-dt,V_threshold,
#     #         t_death=find_stopping_point(dt,pdict,txt_prev,t_prev,save_every_n_frames=10,V_threshold=V_threshold,pid_pair=pid_pair,ds=ds)
#     #         boo=check_atol(pdict,pid_pair,atol,index=-1,ds=ds)
#     #     retval = t_death,min_sigma_max
#     #     retval_lst.append(retval)
#     return retval_lst

##################################################
# TODO: and then births
##################################################
def recursive_zoom_to_birth(pid_pair_lst,in_fn,ds,dt,**kwargs):
    '''TODO(later)'''
    retval_lst=[]
    for pid_pair in pid_pair_lst:
        retval=init_system(in_fn,ds,dt,**kwargs)
        t_birth=find_starting_point(pdict,pid_pair,**kwargs)
        #TODO: after finding starting point,
        # - integrate forward n_steps
        # - return apparent starting point {and return pdict}
        # - reinit at txt_prev / t_min, where t_min is the min(pdict.times)
        #TODO: factor in the tscale/sscale
        #TODO:  find whether to zoom
        while boo:
            system=zoom_system(system)
            t_birth=find_starting_point(pdict,pid_pair,**kwargs)
            boo=check_atol(system,atol)
        retval = t_birth
        retval_lst.append(retval)
    return retval_lst

##################################################
# TODO: make routine executable from command line
##################################################
if __name__ == '__main__':
    pass
