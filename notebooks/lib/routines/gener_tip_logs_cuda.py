import numba, numpy as np, matplotlib.pyplot as plt
from numba import cuda
from pylab import imshow, show
from timeit import default_timer as timer
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#load the libraries
from .. import *
from ..utils.dist_func import *
from ..utils.operari import *
from ..model.minimal_model_cuda import *
from ..utils.utils_jsonio import *
from ..model.minimal_model_cuda import *
from ..utils.ProgressBar import printProgressBar
from ..measure.utils_measure_tips_cpu import fetch_txt_to_tip_dict


def step_forward_2n_times(time_step_kernel,drv,iterations,txt_in,
                         u_new, u_old, v_new, v_old, w_new, w_old,
                         threads, grid,context):
    '''compute the time step in parallel for the correct number of iterations = steps/2.
    total number of memory copies = 2 '''
    n=iterations

    #map input condition to the three input scalar fields
    u_in = np.array(txt_in.astype(np.float64)[...,0])
    v_in = np.array(txt_in.astype(np.float64)[...,1])
    w_in = np.array(txt_in.astype(np.float64)[...,2])

    # #create events for measuring performance
    # start = drv.Event()
    # end = drv.Event()

    #move the data to the GPU
    drv.memcpy_htod(u_old, u_in)
    drv.memcpy_htod(u_new, u_in)
    drv.memcpy_htod(v_old, v_in)
    drv.memcpy_htod(v_new, v_in)
    drv.memcpy_htod(w_old, w_in)
    drv.memcpy_htod(w_new, w_in)

    #call the GPU kernel 2*iterations times (and don't measure performance)
    context.synchronize()
    # start.record()
    for i in range(iterations):
        time_step_kernel(u_new, u_old, v_new, v_old, w_new, w_old, block=threads, grid=grid)
        time_step_kernel(u_old, u_new, v_old, v_new, w_old, w_new, block=threads, grid=grid)
    # end.record()
    context.synchronize()
    # runtime = end.time_since(start)
    # print(f"{iterations*2} time steps took {runtime:.0f} ms.")

    #copy the result from the GPU to Python
    gpu_result_u = np.zeros_like(u_in)
    drv.memcpy_dtoh(gpu_result_u, u_old)
    gpu_result_v = np.zeros_like(v_in)
    drv.memcpy_dtoh(gpu_result_v, v_old)
    gpu_result_w = np.zeros_like(w_in)
    drv.memcpy_dtoh(gpu_result_w, w_old)
    txt_out_gpu = np.stack((gpu_result_u,gpu_result_v,gpu_result_w),axis=2)
    return txt_out_gpu

def get_one_step_map(time_step_kernel,drv,n,
                             u_new, u_old, v_new, v_old, w_new, w_old,
                             threads, grid,context):
    def one_step_map(txt):
        return step_forward_2n_times(time_step_kernel,drv,n,txt,
                             u_new, u_old, v_new, v_old, w_new, w_old,
                             threads, grid,context)
    return one_step_map

def routine_gener_tip_logs_cuda(ic,tmax_sec= 30,tmin_sec=0.,
    printing=True,recording=True,
    diffCoef=0.001,output_time_resolution=2.,context=None):
    '''returns a (pd.DataFrame,np.array) instance
    routine_gener_tip_logs_cuda time evolves initial condition np.array instance, ic, from time tmin_sec
    forward to time tmax_sec via the explicit forward euler method according to the Fenton-Karma model
    (parameter set 8 (wjr's modification)).

    Example Usage:
    context = drv.Device(0).make_context()
    df,txt=routine_gener_tip_logs_cuda(ic,tmax_sec= 30,tmin_sec=0.,
        printing=True,recording=True,
        diffCoef=0.001,output_time_resolution=2.,context=context)
        '''
    height, width, channel_no = ic.shape
    # width  = kwargs['width']
    # height = kwargs['height']

    #load parameters for parameter set 8 for the Fenton-Karma Model
    param_file_name = '/home/timothytyree/Documents/GitHub/care/notebooks/lib/model/param_set_8.json'
    kwargs = read_parameters_from_json(param_file_name)
    dt=0.025
    kwargs['diffCoef']=diffCoef
    kernel_string = get_kernel_string_FK_model(**kwargs, DT=dt,DX=0.025,height=height,width=width)

    ###################################
    # Memory Allocation
    ###################################
    #map initial condition to the three initial scalar fields
    u_initial = np.array(ic.astype(np.float64)[...,0])
    v_initial = np.array(ic.astype(np.float64)[...,1])
    w_initial = np.array(ic.astype(np.float64)[...,2])

    #initializing cuda context
    #initialize PyCuda and get compute capability needed for compilation
    if context is None:
        context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #define how resources are used
    threads = (10,10,1)
    grid = (int(width/10), int(height/10), 1)
    block_size_string = "#define block_size_x 10\n#define block_size_y 10\n"
    #TODO(later): autotune threads and grid using solution in 'dev pycuda'.ipynb

    #don't allocate memory many times for the same task!
    #allocate GPU memory for voltage scalar field
    u_old = drv.mem_alloc(u_initial.nbytes)
    u_new = drv.mem_alloc(u_initial.nbytes)

    #allocate GPU memory for v and w auxiliary fields
    v_old = drv.mem_alloc(v_initial.nbytes)
    v_new = drv.mem_alloc(v_initial.nbytes)
    w_old = drv.mem_alloc(w_initial.nbytes)
    w_new = drv.mem_alloc(w_initial.nbytes)

    #setup thread block dimensions and compile the kernel
    mod = SourceModule(block_size_string+kernel_string)
    time_step_kernel = mod.get_function("time_step_kernel")

    #default observation parameters
    DT=output_time_resolution#2.#ms
    tmax = tmax_sec*10**3#500#nsteps*dt
    nsteps=int(tmax/dt)
    #n equals half of the number of steps between observations of spiral tips
    n = int(DT/2./dt)
    pad = 10
    edge_tolerance = 6
    atol = 1e-10

    # width, height, channel_no = ic.shape
    zero_txt = np.zeros((width, height, channel_no), dtype=np.float64)
    nanstate = [np.nan,np.nan,np.nan]
    ycoord_mesh, xcoord_mesh = np.meshgrid(np.arange(0,width+2*(pad)),np.arange(0,width+2*pad))

    #get kernels jitsu
    txt_to_tip_dict=fetch_txt_to_tip_dict(width,height,DX=0.025,DY=0.025, **kwargs)

    one_step_map=get_one_step_map(time_step_kernel,drv,n,
                                 u_new, u_old, v_new, v_old, w_new, w_old,
                                 threads, grid,context)
    #initialize simulation
    txt = ic.copy()
    tme = 0.
    tip_state_lst = []
    t_values=np.arange(DT,tmax,DT)#np.array([t for t ])
    # integrate in time
    for tme in t_values:
        #step forward 2n times
        txt=one_step_map(txt)
        # tme += 2*n*dt
        if recording:
            dict_out = txt_to_tip_dict(txt, nanstate, zero_txt, xcoord_mesh, ycoord_mesh,
                                pad=pad, edge_tolerance=edge_tolerance, atol=atol,tme=tme)
            tip_state_lst.append(dict_out)
            num_tips = dict_out['n']
            stop_early = (num_tips==0) & (tme>100)
            if stop_early:
                break
        if printing:
            printProgressBar(tme, tmax, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #return tip log as pandas.DataFrame instance
    df = pd.DataFrame(tip_state_lst)
    return df,txt

if __name__=="__main__":
    input_file_name='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ic-in/ic_200x200.001.31.npz'
    try:
        ic = load_buffer(input_file_name)
    except Exception as e1:
        try:
            input_file_name='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ic-out/ic_200x200.001.31.npz'
            ic = load_buffer(input_file_name)
        except Exception as e2:
            print(e1,e2)
            raise ((e1,e2))

    df,txt=routine_gener_tip_logs_cuda(ic,tmax_sec= 30,tmin_sec=0.,
        printing=True,recording=True,
        diffCoef=0.001,output_time_resolution=2.)
    print(df)
