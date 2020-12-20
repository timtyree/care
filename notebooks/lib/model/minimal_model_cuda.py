#!/bin/bash #!/usr/bin/env python3
#############################################################
# CUDA implementation of the Fenton-Karma model
# Programmer: Timothy Tyree
# Date: 8.20.2020
# From: The Rappel Group, University of California, San Diego
#
# Model Definition: F. Fenton and A. Karma, Chaos 8, 20 (1998)
#############################################################
import numpy as np
import pycuda
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

def get_kernel_string_FK_model(DT, round_to=8, **kwargs):
    return get_kernel_string_FK_model_double(DT=DT, round_to=round_to, **kwargs)

def get_kernel_string_FK_model_double(width, height, DX, DT, diffCoef, C_m,
                               tau_pv, tau_v1, tau_v2, tau_pw, tau_mw, tau_d,
                               tau_0, tau_r, tau_si, K, V_sic, V_c, V_v, round_to=8):
    return f"""
// primary model parameters
#define width  {int(width)}
#define height {int(height)}
#define h      {float(DT)}
#define C_m    {float(C_m)}
#define tau_pv {float(tau_pv)}
#define tau_v1 {float(tau_v1)}
#define tau_v2 {float(tau_v2)}
#define tau_pw {float(tau_pw)}
#define tau_mw {float(tau_mw)}
#define tau_d  {float(tau_d)}
#define tau_0  {float(tau_0)}
#define tau_r  {float(tau_r)}
#define tau_si {float(tau_si)}
#define K      {float(K)}
#define V_sic  {float(V_sic)}
#define V_c    {float(V_c)}
#define V_v    {float(V_v)}

// auxiliary model parameters
#define nx width
#define ny height
#define dcoef {float(diffCoef/DX**2):.{round_to}f}

// periodic boundary conditions
__device__ int _pbc ( int q, int wid ) {{
    if (q >= wid) {{
        q = 0;
    }}
    if (q < 0) {{
        q = wid-1;
    }}
    return q;
}}

// heaviside step function
__device__ double step(double thresh, double x)
{{
    return x >= thresh;
}}

// main computation kernel/face shader
__global__ void time_step_kernel(double *u_new, double *u, double *v_new, double *v, double *w_new, double *w) {{
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;
    if (x>=0 && x<nx && y>=0 && y<ny) {{
            double U = u[y*nx+x];
            double V = v[y*nx+x];
            double W = w[y*nx+x];
            double p = step(V_c, U);
            double q = step(V_v, U);
            //WJ's modification of the FK model
            double tau_mv = (1.0 - q) * tau_v2 + q * tau_v1 ;
            ////the unmodified FK model
            //double tau_mv = (1.0 - q) * tau_v1 + q * tau_v2 ;

            // local ion current terms
            double tn  = tanh( K * (U - V_sic)) ;
            double Ifi = -V * p * (U - V_c) * (1.0 - U) / tau_d ;
            double Iso = U * (1.0 - p) / tau_0 + p / tau_r ;
            double Isi = -W * (1.0 + tn ) / (2.0 * tau_si) ;
            double I_sum = Ifi + Iso + Isi ;
            double current_term = -I_sum / C_m ;

            // local transient terms for auxiliary variables
            double dVdt = (1.0 - p) * (1.0 - V) / tau_mv - p * V / tau_pv ;
            double dWdt = (1.0 - p) * (1.0 - W) / tau_mw - p * W / tau_pw ;

            // diffusion term (5-point stencil)
            int up    = _pbc(y+1,height);
            int down  = _pbc(y-1,height);
            int left  = _pbc(x-1, width);
            int right = _pbc(x+1, width);
            double diffusion_term =  dcoef * (
            u[(up)*nx+x]+u[y*nx+right]
            -4.0f*u[y*nx+x]
            +u[y*nx+left]+u[(down)*nx+x]
            );

            //integrate in time
            double dUdt = diffusion_term + current_term ;
            u_new[y*nx+x] = U + h * dUdt;
            v_new[y*nx+x] = V + h * dVdt;
            w_new[y*nx+x] = W + h * dWdt;
    }}
}}
    """

def get_kernel_string_FK_model_single(width, height, DX, DT, diffCoef, C_m,
                               tau_pv, tau_v1, tau_v2, tau_pw, tau_mw, tau_d,
                               tau_0, tau_r, tau_si, K, V_sic, V_c, V_v, round_to=8):
    '''single point precision is supported by lower compute capability gpu's.
    Expect no more than 11 significant figures of accuracy.'''
    return f"""
// primary model parameters
#define width  {int(width)}
#define height {int(height)}
#define h      {float(DT)}
#define C_m    {float(C_m)}
#define tau_pv {float(tau_pv)}
#define tau_v1 {float(tau_v1)}
#define tau_v2 {float(tau_v2)}
#define tau_pw {float(tau_pw)}
#define tau_mw {float(tau_mw)}
#define tau_d  {float(tau_d)}
#define tau_0  {float(tau_0)}
#define tau_r  {float(tau_r)}
#define tau_si {float(tau_si)}
#define K      {float(K)}
#define V_sic  {float(V_sic)}
#define V_c    {float(V_c)}
#define V_v    {float(V_v)}

// auxiliary model parameters
#define nx width
#define ny height
#define dcoef {float(diffCoef/DX**2):.{round_to}f}

// periodic boundary conditions
__device__ int _pbc ( int q, int wid ) {{
    if (q >= wid) {{
        q = 0;
    }}
    if (q < 0) {{
        q = wid-1;
    }}
    return q;
}}

// heaviside step function
__device__ float step(float thresh, float x)
{{
    return x >= thresh;
}}

// main computation kernel/face shader
__global__ void time_step_kernel(float *u_new, float *u, float *v_new, float *v, float *w_new, float *w) {{
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;
    if (x>=0 && x<nx && y>=0 && y<ny) {{
            float U = u[y*nx+x];
            float V = v[y*nx+x];
            float W = w[y*nx+x];
            float p = step(V_c, U);
            float q = step(V_v, U);
            //WJ's modification of the FK model
            float tau_mv = (1.0 - q) * tau_v2 + q * tau_v1 ;
            ////the unmodified FK model
            //float tau_mv = (1.0 - q) * tau_v1 + q * tau_v2 ;

            // local ion current terms
            float tn  = tanh( K * (U - V_sic)) ;
            float Ifi = -V * p * (U - V_c) * (1.0 - U) / tau_d ;
            float Iso = U * (1.0 - p) / tau_0 + p / tau_r ;
            float Isi = -W * (1.0 + tn ) / (2.0 * tau_si) ;
            float I_sum = Ifi + Iso + Isi ;
            float current_term = -I_sum / C_m ;

            // local transient terms for auxiliary variables
            float dVdt = (1.0 - p) * (1.0 - V) / tau_mv - p * V / tau_pv ;
            float dWdt = (1.0 - p) * (1.0 - W) / tau_mw - p * W / tau_pw ;

            // diffusion term (5-point stencil)
            int up    = _pbc(y+1,height);
            int down  = _pbc(y-1,height);
            int left  = _pbc(x-1, width);
            int right = _pbc(x+1, width);
            float diffusion_term =  dcoef * (
            u[(up)*nx+x]+u[y*nx+right]
            -4.0f*u[y*nx+x]
            +u[y*nx+left]+u[(down)*nx+x]
            );

            //integrate in time
            float dUdt = diffusion_term + current_term ;
            u_new[y*nx+x] = U + h * dUdt;
            v_new[y*nx+x] = V + h * dVdt;
            w_new[y*nx+x] = W + h * dWdt;
    }}
}}
    """

#################
# Example Usage #
#################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    from lib.utils_jsonio import *
    # from lib.minimal_model_cuda import *

    #the following might be needed for the kernel_autotuner.
    # drv.init()
    #initialize PyCuda and get compute capability needed for compilation
    context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #load parameters for parameter set 8 for the Fenton-Karma Model
    kwargs = read_parameters_from_json('lib/param_set_8.json')

    #define how resources are used
    width  = kwargs['width']
    height = kwargs['height']
    threads = (10,10,1)
    grid = (int(width/10), int(height/10), 1)
    block_size_string = "#define block_size_x 10\n#define block_size_y 10\n"

    #define the initial conditions
    Vin  = np.array([256*x*(y+1) for x in range(width) for y in range(height)]).reshape((width,height))

    u_initial = Vin.astype(np.float64)
    #initialize auxiliary textures to zero
    v_initial = np.zeros_like(u_initial)
    w_initial = np.zeros_like(u_initial)

    #don't allocate memory many times for the same task!
    #allocate GPU memory for voltage scalar field
    u_old = drv.mem_alloc(u_initial.nbytes)
    u_new = drv.mem_alloc(u_initial.nbytes)

    #allocate GPU memory for v and w auxiliary fields
    v_old = drv.mem_alloc(v_initial.nbytes)
    v_new = drv.mem_alloc(v_initial.nbytes)
    w_old = drv.mem_alloc(w_initial.nbytes)
    w_new = drv.mem_alloc(w_initial.nbytes)

    # explicit time integration up to 5 seconds in steps of size 0.025 ms
    kernel_string = get_kernel_string_FK_model(**kwargs, DT=0.025)
    iterations = 10**5

    #setup thread block dimensions and compile the kernel
    mod = SourceModule(block_size_string+kernel_string)
    time_step_kernel = mod.get_function("time_step_kernel")

    #create events for measuring performance
    start = drv.Event()
    end = drv.Event()

    #move the data to the GPU
    drv.memcpy_htod(u_old, u_initial)
    drv.memcpy_htod(u_new, u_initial)
    drv.memcpy_htod(v_old, v_initial)
    drv.memcpy_htod(v_new, v_initial)
    drv.memcpy_htod(w_old, w_initial)
    drv.memcpy_htod(w_new, w_initial)

    #call the GPU kernel 2*iterations times and measure performance
    context.synchronize()
    start.record()
    for i in range(iterations):
        time_step_kernel(u_new, u_old, v_new, v_old, w_new, w_old, block=threads, grid=grid)
        time_step_kernel(u_old, u_new, v_old, v_new, w_old, w_new, block=threads, grid=grid)
    end.record()
    context.synchronize()
    runtime = end.time_since(start)
    print(f"{iterations*2} time steps took {runtime:.0f} ms.")

    #copy the result from the GPU to Python for plotting
    gpu_result_u = np.zeros_like(u_initial)
    drv.memcpy_dtoh(gpu_result_u, u_old)
    gpu_result_v = np.zeros_like(v_initial)
    drv.memcpy_dtoh(gpu_result_v, v_old)
    gpu_result_w = np.zeros_like(w_initial)
    drv.memcpy_dtoh(gpu_result_w, w_old)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(6,10))
    ax1.imshow(u_initial)
    ax1.set_title("Initial Conditions\n$u_0$")
    ax2.imshow(gpu_result_u)
    ax2.set_title("Final Result From GPU\n$u$")

    ax3.imshow(v_initial)
    ax3.set_title("$v_0$")
    ax4.imshow(gpu_result_v)
    ax4.set_title("$v$")

    ax5.imshow(w_initial)
    ax5.set_title("$w_0$")
    ax6.imshow(gpu_result_w)
    ax6.set_title("$w$")
    plt.show()
