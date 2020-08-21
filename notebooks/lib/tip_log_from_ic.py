#initialize
import numpy as np, pandas as pd, matplotlib.pyplot as plt

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
#beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
#if not 'nb_dir' in globals():
#	nb_dir = os.getcwd()
    
#llvm jit acceleration
from numba import njit

from skimage import measure

#load the libraries
from lib import *
from lib.dist_func import *
from lib.utils_jsonio import *
from lib.operari import *
from lib.get_tips import *
from lib.intersection import *
from lib.tracking import link
from lib.minimal_model_cuda import *
from lib.minimal_model import *
from lib.controller_cuda import *
from lib.utils_measure_tips_cpu import *
from lib.ProgressBar import printProgressBar

##define parameters
#dt = 0.025
##default observation parameters
#n = 50  #half the number of steps between observations of spiral tips
#pad = 10
#edge_tolerance = 6
#atol = 1e-10
#printing=False

def generate_tip_log_from_ic(input_file_name, dt = 0.025,save_every_n_steps = 100, nsteps = 10**7, pad=10, edge_tolerance = 6, atol = 1e-10, printing=False):
    ''' simulates and saves a log of spiral tip data in a file named similar to the initial conditions .npz file located in inpute_file_name.
    nsteps = 10**7 #integrates to 250 seconds when dt = 0.025 ms
    '''
    #load the initial conditions
    n = int(save_every_n_steps/2)
    ic = load_buffer(input_file_name)
    height, width, channel_no = ic.shape

    width, height, channel_no = ic.shape
    zero_txt = np.zeros((width, height, channel_no), dtype=np.float64)
    nanstate = [np.nan,np.nan,np.nan]
    ycoord_mesh, xcoord_mesh = np.meshgrid(np.arange(0,width+2*(pad)),np.arange(0,width+2*pad))



    #load model parameters for parameter set 8 for the Fenton-Karma Model
    param_file_name = '/home/timothytyree/Documents/GitHub/care/notebooks/lib/param_set_8.json'
    model_kwargs = read_parameters_from_json(param_file_name)

    #get the time_step_kernel
    kernel_string = get_kernel_string_FK_model(**model_kwargs, DT=dt)

    #map initial condition to the three initial scalar fields
    u_initial = np.array(ic.astype(np.float64)[...,0])
    v_initial = np.array(ic.astype(np.float64)[...,1])
    w_initial = np.array(ic.astype(np.float64)[...,2])

    #initializing cuda context
    #initialize PyCuda and get compute capability needed for compilation
    context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #define how resources are used
    #width  = kwargs['width']
    #height = kwargs['height']
    threads = (10,10,1)
    grid = (int(width/10), int(height/10), 1)
    block_size_string = "#define block_size_x 10\n#define block_size_y 10\n"

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

    #initialize simulation
    txt = ic.copy()
    tme = 0.
    tip_state_lst = []

    #integrate forward in time
    tmax = nsteps*dt
    while tme <= tmax:
        #step forward 2n times
        txt_in = txt.copy()
        txt = step_forward_2n_times(time_step_kernel,drv,n,txt_in,
                                 u_new, u_old, v_new, v_old, w_new, w_old,
                                 threads, grid, context)
        tme += 2*n*dt

        # measure the tip state
        dict_out = txt_to_tip_dict(txt, nanstate, zero_txt, xcoord_mesh, ycoord_mesh, 
                            pad=pad, edge_tolerance=edge_tolerance, atol=atol, tme=tme)
        tip_state_lst.append(dict_out)
        num_tips = dict_out['n']
        stop_early = (num_tips==0) & (tme>100)
        if stop_early:
            break
        if printing:
            printProgressBar(tme, tmax, prefix = 'Progress:', suffix = 'Complete', length = 50)

    base_dir = '/'.join(os.path.dirname(input_file_name).split('/')[:-1])
    base_save_folder_name = 'ds_5_param_set_8'
    base_save_dir = os.path.join(base_dir, base_save_folder_name)

    #define subfolders
    subfolder_list = ('birth-death-rates', 'trajectories', 'Log')
    data_folder_bdrates = os.path.join(base_save_dir,subfolder_list[0])
    data_folder_traj    = os.path.join(base_save_dir,subfolder_list[1])
    data_folder_log     = os.path.join(base_save_dir,subfolder_list[2])

    data_fn_log = os.path.basename(input_file_name).replace('.npz', '_log.csv')

    #save tip log.csv in the appropriate folder
    df = pd.DataFrame(tip_state_lst)
    os.chdir(data_folder_log)
    df.to_csv(data_fn_log, index=False)

    #move the completed file to ic-out
    completed_ic_fn = input_file_name.replace('ic-in','ic-out')
    os.rename(input_file_name,completed_ic_fn)
    return data_fn_log

if __name__=='__main__':
    #find file interactively
    print("please select a file from within the desired folder.")
    input_file_name = search_for_file()
    generate_tip_log_from_ic(input_file_name)
    


