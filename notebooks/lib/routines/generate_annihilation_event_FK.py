# Compute the spacing between two tips as they annihilate (Fenton-Karma model)
from lib.utils.dist_func import *
from lib.utils.utils_jsonio import *
from lib.model.minimal_model_cuda import *
from lib.measure.utils_measure_tips_cpu import fetch_txt_to_tip_dict
from lib.model.minimal_model import *
from lib.controller.controller_cuda import *

#TODO/DONE?:combine the LR observation method with the FK one_step method
#DONE: get one_step method working in cuda for fk model
#DONE: get tip detection working
#TODO: integrate up until just before death
#TODO: save texture as with the LR model
#TODO: integrate ^this with the existing routine
#TODO(to save time in finding a near death event): once there are 2 tips, save txt_prev

#load initial conditions interactively
#find file interactively
# print("please select a file from within the desired folder.")
# input_file_name = search_for_file()
input_file_name='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ic-in/ic_200x200.001.31.npz'
ic = load_buffer(input_file_name)
tme=0
# plt.imshow(ic)
# plt.title(f"t={tme}")
# plt.show()
height, width, channel_no = ic.shape
#map initial condition to the three initial scalar fields
u_initial = np.array(ic.astype(np.float64)[...,0])
v_initial = np.array(ic.astype(np.float64)[...,1])
w_initial = np.array(ic.astype(np.float64)[...,2])

diffCoef=0.0005
ds=5
dsdpixel=ds/500
dt=0.0005
t=0.;
ds=5.;V_threshold=0.4
#load parameters for parameter set 8 for the Fenton-Karma Model
param_file_name = '/home/timothytyree/Documents/GitHub/care/notebooks/lib/model/param_set_8.json'
kwargs = read_parameters_from_json(param_file_name)
kwargs['diffCoef']=diffCoef
kernel_string = get_kernel_string_FK_model(**kwargs, DT=dt,DX=0.025,height=height,width=width)

# #default observation parameters
# n = 50  #half the number of steps between observations of spiral tips
# nsteps = 10**6
# pad = 10
# edge_tolerance = 6
# atol = 1e-10
# printing=True

width, height, channel_no = ic.shape
zero_txt = np.zeros((width, height, channel_no), dtype=np.float64)
# nanstate = [np.nan,np.nan,np.nan]
# ycoord_mesh, xcoord_mesh = np.meshgrid(np.arange(0,width+2*(pad)),np.arange(0,width+2*pad))

param_fn=param_file_name
param_dir = os.path.join(nb_dir,'lib/model')
param_dict = json.load(open(os.path.join(param_dir,param_fn)))
if diffCoef is not None:
    param_dict['diffCoef']=diffCoef

get_time_step=fetch_get_time_step(width,height,DX=dsdpixel,DY=dsdpixel,**param_dict)

# print(kernel_string)

#initializing cuda context
#initialize PyCuda and get compute capability needed for compilation
context = drv.Device(0).make_context()
devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

#define how resources are used
# width  = kwargs['width']
# height = kwargs['height']
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

one_step_map=get_one_step_map(time_step_kernel, drv, u_new, u_old, v_new, v_old, w_new, w_old, threads, grid, context)

# txt=load_buffer(txt_fn)
# inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
# width,height=txt.shape[:2]
print(txt.shape)
one_step,comp_distance,comp_dict_tips=init_methods(width,height,ds,dt,V_threshold=V_threshold,jump_threshold=40)
comp_dict_topo_full_color=comp_dict_tips


#initialize simulation
txt = ic.copy()
tme = 0.
nsteps=1000

#find the termination tmie
for m in range(100):
    txt=one_step_map(txt,nsteps=nsteps)
    tme+=2*dt*nsteps

#reidentify the tips to be tracked
dtxt_dt = zero_txt.copy()
get_time_step(txt, dtxt_dt)
img=txt[...,0];dimgdt=dtxt_dt[...,0]
# img=inVc[...,0];dimgdt=dVcdt[...,0]
dict_tips=comp_dict_tips(img, dimgdt, t, txt)
pdict=ParticlePBCDict(dict_tips=dict_tips, width=width, height=width)#, **kwargs)
ntips=len(dict_tips['x'])

while ntips>0:
    t_prev=t;txt_prev=txt.copy()
    # for m in range(50):
    #     txt=one_step_map(txt,nsteps=nsteps)
    #     tme+=2*dt*nsteps
    txt=one_step_map(txt,nsteps=nsteps)
    tme+=2*dt*nsteps
    #reidentify the tips to be tracked
    dtxt_dt = zero_txt.copy()
    get_time_step(txt, dtxt_dt)
    img=txt[...,0];dimgdt=dtxt_dt[...,0]
    # img=inVc[...,0];dimgdt=dVcdt[...,0]
    dict_tips=comp_dict_tips(img, dimgdt, tme, txt)
    pdict.merge_dict(dict_tips)
    ntips=len(dict_tips['x'])
    print(f"ntips={ntips:.0f}, time={tme:.2f}.",end='\r')



#test the V_threshold value
j=1
V_threshold=0.4
level1 = V_threshold
t=-999
level2 = 0.
one_step,comp_distance,comp_dict_tips=init_methods(width,height,ds,dt,V_threshold=V_threshold,jump_threshold=40)

#reidentify the tips to be tracked
dtxt_dt = zero_txt.copy()
get_time_step(txt_prev, dtxt_dt)
img=txt_prev[...,0];dimgdt=dtxt_dt[...,0]
# compute_all_spiral_tips= get_compute_all_spiral_tips(mode='simp',width=width,height=height)
# dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2)#,width=width,height=height)
dict_tips=comp_dict_tips(img, dimgdt, tme, txt_prev)
# print(len(list(dict_out['x'])))
# fig=show_buffer_LR(txt)

ntips=len(dict_tips['x'])
print(f"ntips={ntips:.0f}, time={tme:.2f}.",end='\r')
# plt.imshow(img,cmap='gray')
# plt.scatter(dict_tips['x'],dict_tips['y'],s=300,c='yellow',marker='*')
# plt.show()
