from .. import *
import numpy as np
from ..controller.controller_LR import get_one_step_explicit_synchronous_splitting


def zoom_system(txt,pdict):
    txt=zoomin_txt(txt)
    pdict=pdict.zoom_to_double()
    return txt,pdict

def compute_last_sigma_max(pdict,ds=5.):
    pid_lst=pdict.get_alive_particles()
    last_particle=pdict[pid_lst[0]]
    scale=ds/last_particle.width #cm/pixel
    sigma_max=scale*last_particle['lesser_arclen'][-1]
    return sigma_max


def find_stopping_point(dt,pdict,txt_prev,t_prev,save_every_n_frames=1,V_threshold=-50.,pid_pair=(0,1),ds=5.):
    width,height=txt_prev.shape[:2]
    # get_one_step at this dt,ds
    # comp_dict_topo_simple=get_comp_dict_topo_simple(width=width,height=height)
    __, __, one_step = get_one_step_explicit_synchronous_splitting(nb_dir,
                                                                   dt=dt,width=width,height=height,
                                                                   ds=5.,diffCoef=0.001,Cm=1.0)
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

        #compute particles that are present
        in_to_out=pdict.sort_particles_indices(dict_topo)#, search_range=40.)
        pid_lst_living=in_to_out.values
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


##################################################
# TODO: make routine executable from command line
##################################################
if __name__ == '__main__':
    pass
