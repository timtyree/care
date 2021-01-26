from .. import *
from ..model.LR_model_optimized import *
from ..utils.stack_txt_LR import stack_txt, unstack_txt


def get_one_step_map_w_Istim(nb_dir,dt,**kwargs):
    '''returns dt,one_step_map'''
    dt, arr39, one_step_method = get_one_step_explicit_synchronous_splitting_w_Istim(nb_dir,dt,**kwargs)
    @njit
    def one_step_map_w_Istim(txt, txt_Istim):
        #unstack txt
        inVc,outVc,inmhjdfx,outmhjdfx,dVcdt = unstack_txt(txt)
        #integrate by dt
        one_step_method(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,txt_Istim)
        # t+=dt
        #stack txt
        # txt_nxt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        # return txt_nxt
    return dt,one_step_map_w_Istim


def get_one_step_explicit_synchronous_splitting_w_Istim(nb_dir,dt=0.01,width=200,height=200,ds=5.,diffCoef=0.001,Cm=1.,**kwargs):
    '''returns dt, arr39, one_step_explicit_synchronous_splitting_w_Istim
    precomputes lookup table, arr39 and returns a jit compiling one_step method
    '''
    #precompute lookup table
    arr39=get_arr39(dt,nb_dir)
    v_values=arr39[:,0]
    lookup_params=get_lookup_params(v_values,dv=0.1)
    comp_dVcdt=get_comp_dVcdt(width=width, height=height, diffCoef=diffCoef, ds=ds, Cm=Cm)

    @njit
    def one_step_explicit_synchronous_splitting_w_Istim(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,txt_Istim):
        '''
        for each pixel:
            advances V and Ca_i by dt/2 for each pixel using forward euler integration
        and then,
        for each pixel:
            advances gating variables using the exact flow map resulting from V
            advances V and Ca_i by dt/2 for each pixel using forward euler integration
        enforces agreement between inVc and outVc and between inmhjdfx and outmhjdfx (to avoid any confusion)
        '''
        for x in range(width):
            for y in range(height):
                #extract local variables
                Vc = inVc[x,y]; V=Vc[0]
                inCgate = inmhjdfx[x,y]

                #parse the row linearly interpolated from lookup table
                arr_interp=lookup_params(V,arr39)
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',

                #half step voltage and calcium
                dVcdt_val=comp_dVcdt(inVc, x, y, inCgate, IK1T, x1)
                #include the effect of external stimulus current
                Istim = txt_Istim[x,y]
                dVcdt_val[0]-=Istim/Cm
                #record the result
                outVc_val=Vc+0.5*dt*dVcdt_val
                outVc[x,y]=outVc_val.copy()
        for x in range(width):
            for y in range(height):
                #parse the row linearly interpolated from lookup table with updated voltage
                inCgate  = inmhjdfx[x,y]
                outCgate = outmhjdfx[x,y]
                Vc = outVc[x,y]; V  = Vc[0]
                arr_interp=lookup_params(V,arr39)
                # x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',

                #full step the gating variables of step size dt (dt is encoded in arr39)
                comp_exact_next_gating_var(inCgate,outCgate,arr_interp)
                inmhjdfx[x,y]=outCgate.copy()
                outmhjdfx[x,y]=outCgate.copy()

                #half step voltage and calcium
                dVcdt_val=comp_dVcdt(outVc, x, y, outCgate, IK1T, x1)
                #include the effect of external stimulus current
                Istim = txt_Istim[x,y]
                dVcdt_val[0]-=Istim/Cm
                #record the result
                outVc_val=Vc+0.5*dt*dVcdt_val
                outVc[x,y]=outVc_val.copy()
                inVc[x,y]=outVc_val.copy()

                #compute the current voltage/sodium flow map
                Vc = outVc[x,y]; V  = Vc[0]
                arr_interp=lookup_params(V,arr39)
                # x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',
                dVcdt_val=comp_dVcdt(outVc, x, y, outCgate, IK1T, x1)
                #record rate of change of voltage and calcium current
                dVcdt[x,y]=dVcdt_val.copy()

                #save texture to output
                # txt_nxt[x,y]=stack_pxl(outVc_val,outVc_val,outCgate,outCgate,dVcdt_val)

        # #copy out to in
        # inmhjdfx=outmhjdfx.copy()
        # outVc=inVc.copy()
        # np.stack(*(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)).T
    return dt, arr39, one_step_explicit_synchronous_splitting_w_Istim



def get_one_step_explicit_synchronous_splitting(nb_dir,dt=0.01,width=200,height=200,ds=5.,diffCoef=0.001,Cm=1.,**kwargs):
    '''returns dt, arr39, one_step_explicit_synchronous_splitting
    precomputes lookup table, arr39 and returns a jit compiling one_step method
    '''
    #precompute lookup table
    arr39=get_arr39(dt,nb_dir)
    v_values=arr39[:,0]
    lookup_params=get_lookup_params(v_values,dv=0.1)
    comp_dVcdt=get_comp_dVcdt(width=width, height=height, diffCoef=diffCoef, ds=ds, Cm=Cm)

    @njit
    def one_step_explicit_synchronous_splitting(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt):
        '''
        for each pixel:
            advances V and Ca_i by dt/2 for each pixel using forward euler integration
        and then,
        for each pixel:
            advances gating variables using the exact flow map resulting from V
            advances V and Ca_i by dt/2 for each pixel using forward euler integration
        enforces agreement between inVc and outVc and between inmhjdfx and outmhjdfx (to avoid any confusion)
        '''
        for x in range(width):
            for y in range(height):
                #extract local variables
                Vc = inVc[x,y]; V=Vc[0]
                inCgate = inmhjdfx[x,y]

                #parse the row linearly interpolated from lookup table
                arr_interp=lookup_params(V,arr39)
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',

                #half step voltage and calcium
                dVcdt_val=comp_dVcdt(inVc, x, y, inCgate, IK1T, x1)
                outVc_val=Vc+0.5*dt*dVcdt_val
                outVc[x,y]=outVc_val.copy()
        for x in range(width):
            for y in range(height):
                #parse the row linearly interpolated from lookup table with updated voltage
                inCgate  = inmhjdfx[x,y]
                outCgate = outmhjdfx[x,y]
                Vc = outVc[x,y]; V  = Vc[0]
                arr_interp=lookup_params(V,arr39)
                # x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',

                #full step the gating variables of step size dt (dt is encoded in arr39)
                comp_exact_next_gating_var(inCgate,outCgate,arr_interp)
                inmhjdfx[x,y]=outCgate.copy()
                outmhjdfx[x,y]=outCgate.copy()

                #half step voltage and calcium
                dVcdt_val=comp_dVcdt(outVc, x, y, outCgate, IK1T, x1)
                outVc_val=Vc+0.5*dt*dVcdt_val
                outVc[x,y]=outVc_val.copy()
                inVc[x,y]=outVc_val.copy()

                #compute the current voltage/sodium flow map
                Vc = outVc[x,y]; V  = Vc[0]
                arr_interp=lookup_params(V,arr39)
                # x_infty,tau_x,m_infty,tau_m,h_infty,tau_h,j_infty,tau_j,d_infty,tau_d,f_infty,tau_f,IK1T,x1,e1,em,eh,ej,ed,ef=arr_interp[1:]
                IK1T=arr_interp[13]    # 'xttab',
                x1=arr_interp[14]    # 'x1',
                dVcdt_val=comp_dVcdt(outVc, x, y, outCgate, IK1T, x1)
                #record rate of change of voltage and calcium current
                dVcdt[x,y]=dVcdt_val.copy()

        # #copy out to in
        # inmhjdfx=outmhjdfx.copy()
        # outVc=inVc.copy()
        # np.stack(*(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)).T
    return dt, arr39, one_step_explicit_synchronous_splitting

def get_one_step_map(nb_dir,dt,**kwargs):
    '''returns dt,one_step_map'''
    dt, arr39, one_step_method = get_one_step_explicit_synchronous_splitting(nb_dir,dt,**kwargs)
    @njit
    def one_step_map(txt):
        #unstack txt
        inVc,outVc,inmhjdfx,outmhjdfx,dVcdt = unstack_txt(txt)
        #integrate by dt
        one_step_method(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        # t+=dt
        #stack txt
        txt_nxt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        return txt_nxt
    return dt,one_step_map


# def get_one_step_kernel_LR_forward_euler(ds = 0.015, width =200, height=200, Cm=1., diffCoef=0.001,
#                        Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,method='njit'):
#     if method=='njit':
#         njitsu = njit
#     if method=='cuda':
#         import numba.cuda.njit as njitsu
#     comp_rate_of_voltage_change_at_pixel_LR=get_comp_rate_of_voltage_change_at_pixel_LR(ds = ds, width = width, height = height, Cm=Cm, diffCoef=diffCoef,
#                                                                         Na_i = Na_i, Na_o = Na_o, K_i  = K_i, K_o  = K_o, Ca_o = Ca_o,method=method)
#     comp_exact_flow_map_gating_variables=get_comp_exact_flow_map_gating_variables(method=method)
#     # pbc=get_pbc(width,height)

#     @njitsu
#     def compute_one_step_kernel_LR_forward_euler(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,dt):
#         for x in range(width):
#             for y in range(height):
#                 # x,y = pbc(x,y)
#                 C = inVc[x,y]
#                 inCgate = inmhjdfx[x,y]
#                 outCgate = outmhjdfx[x,y]
#                 V = C[0]
#                 #update gating variables with next value
#                 comp_exact_flow_map_gating_variables(inCgate, outCgate, V, dt)
#                 #compute current at pixel using gating variables evaluated according to the implicit midpoint rule
#                 Cgate=0.5*inCgate+0.5*outCgate
#                 dVcdt_val = comp_rate_of_voltage_change_at_pixel_LR(inVc, C, Cgate, x, y)
#                 dVcdt[x,y] = dVcdt_val
#                 outVc[x,y] = C + dt*dVcdt_val
#     return compute_one_step_kernel_LR_forward_euler

# def get_multi_step_forward_euler(ds = 0.015, width =200, height=200, Cm=1., diffCoef=0.001,
#                        Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,method='njit'):
#     '''Uses a novel operating splitting method that is agnostic to whether each time step is the first/last in a sequence'''
#     if method=='njit':
#         njitsu = njit
#     if method=='cuda':
#         import numba.cuda.njit as njitsu
#     one_step_kernel_LR=get_one_step_kernel_LR_forward_euler(ds = 0.015, width =200, height=200, Cm=1., diffCoef=0.001,
#                        Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8, method='njit')
#     @njitsu
#     def multi_step_forward_euler(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,dt,half_num_steps):
#         '''advances 2*half_num_steps time steps, each of size dt.'''
#         for stepnum in range(half_num_steps):
#             one_step_kernel_LR(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt,dt)
#             one_step_kernel_LR(outVc,inVc,outmhjdfx,inmhjdfx,dVcdt,dt)
#         # return inVc,outVc,inmhjdfx,outmhjdfx,dVcdt
#     return multi_step_forward_euler

# ##############################
# # Deprecated yet functional
# ###############################
# def get_compute_dtxtdt(ds = 0.015, width =200, height=200, Cm=1., diffCoef=0.001,
#                        Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,method='njit'):
#     if method=='njit':
#         njitsu = njit
#     if method=='cuda':
#         import numba.cuda.njit as njitsu
#     comp_rate_of_change_at_pixel_LR=get_comp_rate_of_change_at_pixel_LR(ds = ds, width = width, height = height, Cm=Cm, diffCoef=diffCoef,
#                                                                         Na_i = Na_i, Na_o = Na_o, K_i  = K_i, K_o  = K_o, Ca_o = Ca_o,method=method)
#     @njitsu
#     def compute_dtxtdt(txt,out):
#         for x in range(width):
#             for y in range(height):
#                 out[x,y] = comp_rate_of_change_at_pixel_LR(txt, x, y)
#     return compute_dtxtdt

# def get_compute_dVcdt(ds = 0.015, width =200, height=200, Cm=1., diffCoef=0.001,
#                        Na_i = 18, Na_o = 140, K_i  = 145, K_o  = 5.4, Ca_o = 1.8,method='njit'):
#     if method=='njit':
#         njitsu = njit
#     if method=='cuda':
#         import numba.cuda.njit as njitsu
#     comp_rate_of_voltage_change_at_pixel_LR=get_comp_rate_of_voltage_change_at_pixel_LR(ds = ds, width = width, height = height, Cm=Cm, diffCoef=diffCoef,
#                                                                         Na_i = Na_i, Na_o = Na_o, K_i  = K_i, K_o  = K_o, Ca_o = Ca_o,method=method)
#     @njitsu
#     def compute_dVcdt(inVc,outVc,inmhjdfx):
#         for x in range(width):
#             for y in range(height):
#                 outVc[x,y] = comp_rate_of_voltage_change_at_pixel_LR(inVc, inmhjdfx, x, y)
#     return compute_dVcdt

# # def get_pbc(width,height):
# #     @njit
# #     def pbc(S,x,y):
# #         '''S=texture with size 512,512,3
# #         (x, y) pixel coordinates of texture with values 0 to 1.
# #         tight boundary rounding is in use.'''
# #         if ( x < 0  ):              # // Left P.B.C.
# #             x = width - 1
# #         elif ( x > (width - 1) ):   # // Right P.B.C.
# #             x = 0
# #         if( y < 0 ):                # //  Bottom P.B.C.
# #             y = height - 1
# #         elif ( y > (height - 1)):   # // Top P.B.C.
# #             y = 0
# #         return x,y

# def get_one_step_forward_euler(txt,compute_dtxtdt,V_max=100.,method='njit'):
#     '''Solution via operator splitting method.
#     Calcium and voltage are advanced by forward euler integration
#     gating variables are advanced by their exact flow map.
#     Exception ZeroDivisionError: division by zero may be thrown if V becomes larger than 10000. mV.  Consider lowering time step for stability.'''
#     if method=='njit':
#         njitsu = njit
#     if method=='cuda':
#         import numba.cuda.njit as njitsu
#     zero_txt = np.zeros_like(txt)
#     @njitsu
#     def one_step_forward_euler(txt, dt):
#         #compute next values for gating variables
#         dtxt_dt=zero_txt.copy()
#         compute_dtxtdt(txt,out=dtxt_dt)
#         txt += dt*dtxt_dt

#         # #limit voltage to the maximum voltage
#         # boo=txt[...,0]>V_max
#         # if boo.any():
#         #     txt[...,0][boo]=V_max

#     return one_step_forward_euler
