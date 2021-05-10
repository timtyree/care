import numpy as np, pandas as pd
#Programmer: Tim Tyree
#Date: 5.10.2021
#Group: Rappel

############
#Conventions
############
# $$
# \text{Let   } \varphi_1\equiv\sin^{-1}\big(\widehat{\mathbf{x}_2-\mathbf{x}_1}\;\times\;\hat{\mathbf{a}}_1\big),
# $$

# $$
# \text{and let   } \varphi_2\equiv\sin^{-1}\big(\widehat{\mathbf{x}_1-\mathbf{x}_2}\;\times\;\hat{\mathbf{a}}_2\big).
# $$

#######
#Module
#######
def compute_phases_between(d1,d2,dict_activation_front,field='lesser_xy_values'):
    '''computes the phases between particle 1 and particle 2 in units of radians.
    returns range between particles in units of pixels'''
    # compute displacement unit vector from tip 1 to tip 2
    xy1_values=np.array(list(zip(d1['x'],d1['y'])))
    xy2_values=np.array(list(zip(d2['x'],d2['y'])))
    xy2_minus_xy1_values=xy2_values-xy1_values
    range_values=np.linalg.norm(xy2_minus_xy1_values, axis=1)
    x2_minus_x1_hat_values=xy2_minus_xy1_values[:,0]/range_values
    y2_minus_y1_hat_values=xy2_minus_xy1_values[:,1]/range_values
    xy2_minus_xy1_hat_values=np.array(list(zip(x2_minus_x1_hat_values,y2_minus_y1_hat_values)))
    

    #compute directions of activation fronts.  store as pandas.DataFrame.
    daf=dict_activation_front
    #     daf.keys()
    #time values
    t1_values=d1['t'].values
    t_values=np.array(daf['t'])[1:]

    xy_values_lst=daf[field][1:]
    phi1_lst=[];phi2_lst=[]
    for i in range(len(xy_values_lst)):
        dx1dx2_hat=xy2_minus_xy1_hat_values[i]
        # print(t1_values[i])

        #TODO(if naive is ugly...): try moving avg of first j contour points for the ith observation time

        xy_values=xy_values_lst[i]

        #TODO: compute a1_hat and a2_hat
        a1=xy_values[1]-xy_values[0]
        # xy_values[2]-xy_values[1]
        # xy_values[3]-xy_values[2]
        # xy_values[4]-xy_values[3]
        a1_hat=a1/np.linalg.norm(a1)

        a2=xy_values[-2]-xy_values[-1]
        a2_hat=a2/np.linalg.norm(a2)

        #TODO(later, to scale method): convert all subtraction operations to explicitely enforce pbc...
        # print(t_values[i])

#         #assert we're comparing the right times
#         print ((i, t_values[i] , t1_values[i]))
#         assert ( t_values[i] == t1_values[i])

        phi1=np.arcsin(np.cross(dx1dx2_hat,a1))
        phi2=np.arcsin(np.cross(-1.*dx1dx2_hat,a2))

        phi1_lst.append(phi1)
        phi2_lst.append(phi2)
    
    phi1_values=np.array(phi1_lst)
    phi2_values=np.array(phi2_lst)

    #decide to make 90 degrees positive.
    boo=np.isnan(phi1_values)
    phi1_values[boo]=np.pi/2.
    boo=np.isnan(phi2_values)
    phi2_values[boo]=np.pi/2.
    
    return range_values,phi1_values,phi2_values


def comp_relative_phase(phi1_values,phi2_values):
    phi_sum_values=phi2_values+phi1_values
    phi_diff_values=phi2_values-phi1_values
    return phi_sum_values, phi_diff_values

##############
#Example Usage
##############
def simulate_pdict_example(dt=0.001,V_threshold=-50.):
    txt_fn=f'{nb_dir}/Data/test_data/ic008.33_t_218.8.npz'
    t=218.8;ds=5.;
    txt=load_buffer(txt_fn)
    inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
    width,height=txt.shape[:2]
    print(txt.shape)
    one_step,comp_distance,comp_dict_tips=init_methods(width,height,ds,dt,V_threshold=V_threshold,jump_threshold=40)
    comp_dict_topo_full_color=comp_dict_tips
    #reidentify the tips to be tracked
    img=inVc[...,0];dimgdt=dVcdt[...,0]
    dict_tips=comp_dict_tips(img, dimgdt, t, txt)
    pdict=ParticlePBCDict(dict_tips=dict_tips, width=width, height=width)#, **kwargs)
    t_prev=t;txt_prev=txt.copy()

    #visualize token death system
    x_values=np.array(dict_tips['x'])[:-2]
    y_values=np.array(dict_tips['y'])[:-2]
    c_values=np.array(dict_tips['pid'])[:-2]
    # fig = ShowDomain(img,dimgdt,x_values,y_values,c_values,V_threshold,t,inch=6,
    #                  fontsize=16,vmin_img=-85.,vmax_img=35.,area=25,
    #                  frameno=None,#frameno,
    #                  save_fn=None,#save_fn,
    #                  save_folder=None,#save_folder,
    #                  save=False,#True,
    #                  annotating=True,
    #                  axis=[0,img.shape[0],0,img.shape[1]])

    #better method: take more data! (after condensing data to a simple readout)
    #DONE/DONT(later): look for more reliable way to identify activation fronts... use consistency of position over time??
    # HINT: consider looking at mean gating variables from the comp_dict_topo_full_color
    #TODO: linearly record data for ^these spiral tips at a high sampling rate (and fine spatial resolution)

    #TODO(brainwarmer): check Slides for whether tips move along dVdt levelsets or along V levelsets
    #TODO: load/plot system
    #TODO: compute each of the the final scalar values needed for the following...
    #TODO: test angular difference between cartesion acceleration and acceleration in the direction of the activation front versus time
    #TODO: linearly track lesser_arclen of these two death events on a dt=0.001 ms timescale fixed at the basic subpixel resolution
    #TODO: test proposition that lesser_arclen always drops shortly annihilating, perhaps on the 0.01~0.02 (ms?) timescale.  
    # ^This would support the mechanism of annihilation involving the connection of activation fronts/strings with some tension to contract
    # ^This would support using a model of spiral tip dynamics along an activation front to inform our reaction rate calculations
    # HINT: consider rate = 1/expected_time_to_death, where the arclength behaves in a predictable way, i.e.
    # dsigma_max_dt=foo(sigma_max;relative phase?)

    # from inspect import getsource
    # print ( getsource(pdict.record_tips_return_txt))
    # pdict.record_tips_return_txt?
    ntips=len(dict_tips['x'])
    assert(ntips>0)

    inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
    frameno=0
    change_time=0.
    duration =0.03
    save_every_n_frames=1
    while change_time<=duration:
    # while ntips>0:
        frameno+=1
        t+=dt
        change_time+=dt
        one_step(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        if frameno % save_every_n_frames == 0:
            dict_tips=comp_dict_tips(img, dimgdt, t, txt)
            pdict.merge_dict(dict_tips)
            ntips=len(dict_tips['x'])
            print(f"saved at time {t:.3f} ms.",end='\r')

    txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)

    return pdict


if __name__=='__main__':
    pdict=simulate_pdict_example()#dt=0.001,V_threshold=-50.)
    #compute the pdict
    df, dict_greater_dict, dict_lesser_dict=pdict.separate_data_to_pandas()

    #extract the relevant particles
    d1=df[df.pid==float(pid_pair[0])].copy()
    d2=df[df.pid==float(pid_pair[1])].copy()
    dict_activation_front=dict_lesser_dict[pid_pair[0]]

    #compute the relative phases of spiral tips
    phi1_values,phi2_values,range_values=compute_phases_between(d1,d2,dict_activation_front)
    phi_sum_values, phi_diff_values=comp_relative_phase(phi1_values,phi2_values)
    i=27

    #print results
    print(f"phi1   , phi2     = {phi1:.3f},{phi2:.3f} at time {t_values[i]:.3f}.")
    print(f"phi_sum, phi_diff = {phi_sum:.3f},{phi_diff:.3f} at time {t_values[i]:.3f}.")
