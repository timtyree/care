#controller_cuda.py
#TODO: move example usage from tip_log_from_ic.py to here
import numpy as np

def step_forward_2n_times(time_step_kernel,drv,n,txt_in,
                         u_new, u_old, v_new, v_old, w_new, w_old,
                         threads, grid, context):
    '''compute the time step for the correct number of iterations = steps/2'''
    iterations = n

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
