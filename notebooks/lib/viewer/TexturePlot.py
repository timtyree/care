import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters
from .. import *
# from lib.get_tips import *
# from lib.intersection import *
# from lib import *
from ..model.minimal_model import *
import matplotlib.cm as cm


def show_buffer_LR(txt,figsize=(9,9),rgb_channels=(0,-2,1)):
	"""Visualize the buffer (plain)"""
	# print(np.max(txt[...,0]))
	img=txt[...,rgb_channels].copy()#chnl]#0,1,V,Ca_i,INa,IK
	mn=np.min(img[...,0]);mx=np.max(img[...,0])
	img[...,0]=(img[...,0]-mn)/(mx-mn)
	# mn=np.min(img[...,1]);mx=np.max(img[...,1])
	# img[...,1]=(img[...,1]-mn)/(mx-mn)
	mn=np.min(img[...,2]);mx=np.max(img[...,2])
	img[...,2]=(img[...,2]-mn)/(mx-mn)
	fig,ax=plt.subplots(figsize=figsize)
	ax.imshow(img)#,cmap='gray',vmin=-80, vmax=15)
	ax.axis('off')
	return fig

def plot_buffer(img_nxt, img_inc, contours_raw, contours_inc, tips,
	figsize=(15,15), max_marker_size=800, lw=2,
	color_values = None):
	'''computes display data; returns fig.'''
	#plot figure
	fig, ax = plt.subplots(1,figsize=figsize)
	ax.imshow(img_nxt,cmap='Reds', vmin=0, vmax=1)
	ax.axis('off')

	#plot contours, if any.  type 1 = contours_raw (blue), type 2 = contours_inc (green)
	for n, contour in enumerate(contours_inc):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='g', zorder=2)
	for n, contour in enumerate(contours_raw):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='b', zorder=2)

	#plot tips, if any
	s1_values, s2_values, y_values, x_values, states_nearest, states_interpolated_linear, states_interpolated_cubic = tips
	#     if len(n_values)>0:
	if color_values is None:
		for j in range(len(x_values)):
			ax.scatter(x = x_values[j], y = y_values[j], c='yellow', s=int(max_marker_size/(j+1)), zorder=3, marker = '*')
		pass
	else:
		vmin = 0.#np.min(color_values)
		vmax = 1.#np.max(color_values)
		cmap = plt.get_cmap('cividis')
		for j in range(len(x_values)):
		    ax.scatter(x = x_values[j], y = y_values[j], c=color_values[j], s=int(max_marker_size/(j+1)),
		               zorder=3, marker = '*', cmap=cmap, vmin=vmin, vmax=vmax)
	return fig

def show_buffer (txt,
	sigma       = 3.,#5#1
	threshold   = 0.8,#9#0.95
	V_threshold = 0.7,
	describe_it=False):#0.5

	# check all the functions work/ compile the needed functions
	if describe_it:
		describe(txt)#optional printing
	width, height, channel_no = txt.shape
	zero_txt = np.zeros((width, height, channel_no), dtype=np.float64)
	dtexture_dt = zero_txt.copy()
	get_time_step(txt, dtexture_dt)

	#calculate contours and tips
	img_nxt = txt[..., 0]
	img_inc = ifilter(dtexture_dt[..., 0])  #mask of instantaneously increasing voltages
	img_inc = filters.gaussian(img_inc,sigma=sigma, mode='wrap')
	contours_raw = measure.find_contours(img_nxt, level=V_threshold,fully_connected='low',positive_orientation='low')
	contours_inc = measure.find_contours(img_inc, level=threshold)#,fully_connected='low',positive_orientation='low')
	tips  = get_tips(contours_raw, contours_inc)
	n_old = count_tips(tips[1])

	#bluf
	# print(f"\n number of type 1 contour = {len(contours_raw)},\tnumber of type 2 contour = {len(contours_inc)},")
	print(f"the number of tips are {n_old}.")
	# print(f"""the topological tip state:{tips[0]}""")
	# print(f"""x position of tips: {tips[1]}""")
	# print(f"""y position of tips: {tips[2]}""")

	n_lst, x_lst, y_lst = get_tips(contours_raw, contours_inc)
	tip_states = {'n': n_lst, 'x': x_lst, 'y': y_lst}

	fig = plot_buffer(img_nxt, img_inc, contours_raw, contours_inc, tips,
					  figsize=(5,5),max_marker_size=200, lw=1);
	# fig.show()
	return fig


####################################
# Elementary texture viewing
####################################
def describe_texture(txt,n):
	print(f"""for channel {n},
	max value: {np.max(txt)}
	min value: {np.min(txt)}
	mean value: {np.mean(txt)}
	""")

def describe(txt):
	'''Example usage:
	describe_txt(txt)'''
	describe_texture(txt[..., 0],0)
	describe_texture(txt[..., 1],1)
	describe_texture(txt[..., 2],2)

def display_texture(txt, vmins=(0, 0, 0), vmaxs=(1, 1, 1), title0 = 'channel 0', title1 = 'channel 1', title2 = 'channel 2'):
	'''Example usage:
	# txt = np.load('Data/buffer_test_error.npy')
	# dtexture_dt = np.zeros((width, height, channel_no), dtype = np.float64)
	# get_time_step(txt , dtexture_dt)
	# display_texture(txt, vmins=(0,0,0),vmaxs=(1,1,1))
	# display_texture(dtexture_dt, vmins=(0,0,0),vmaxs=(1,1,1))'''
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
	ax1.imshow(txt[...,0], cmap='Reds', vmin=vmins[0], vmax=vmaxs[0])
	ax2.imshow(txt[...,1], cmap='Reds', vmin=vmins[1], vmax=vmaxs[1])
	ax3.imshow(txt[...,2], cmap='Reds', vmin=vmins[2], vmax=vmaxs[2])
	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax1.set_title(title0)
	ax2.set_title(title1)
	ax3.set_title(title2)
	return fig, (ax1, ax2, ax3)

####################################
# Axis Painting Operations
####################################
def plot_texture(txt, vmins, vmaxs, ax1, ax2, ax3, title0 = 'channel 0', title1 = 'channel 1', title2 = 'channel 2', fontsize=18):
	'''Example usage:
	# txt = np.load('Data/buffer_test_error.npy')
	# dtexture_dt = np.zeros((width, height, channel_no), dtype = np.float64)
	# get_time_step(txt , dtexture_dt)
	# plt_texture(txt, vmins=(0,0,0),vmaxs=(1,1,1))'''
	ax1.imshow(txt[...,0], cmap='Reds', vmin=vmins[0], vmax=vmaxs[0], label='voltage/channel 0')
	ax2.imshow(txt[...,1], cmap='Reds', vmin=vmins[1], vmax=vmaxs[1], label='fast_var/channel 1')
	ax3.imshow(txt[...,2], cmap='Reds', vmin=vmins[2], vmax=vmaxs[2], label='slow_var/channel 2')
	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax1.set_title(title0, fontsize=fontsize)
	ax2.set_title(title1, fontsize=fontsize)
	ax3.set_title(title2, fontsize=fontsize)
	return ax1, ax2, ax3


def plot_contours(ax, contours_raw, contours_inc, color_raw='green', color_inc='blue', lw=2):
	'''texture is a 1 channelled 2D image'''
	booa = len(contours_raw)>0;boob = len(contours_inc)>0;
	if booa:
		for n, contour in enumerate(contours_raw):
			ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c=color_raw, zorder=1)
	if boob:
		for n, contour in enumerate(contours_inc):
			ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c=color_inc, zorder=2)
	return ax

def plot_tips(ax, n_list, x_list, y_list, color_tips='white'):
	'''ax is matplotlib axis. ._list is a list of tip features.
	TODO: color map plot_tips by n_list'''
	boo = len(tips)>0;
	if boo:
		for n, contour in enumerate(contours_raw):
			ax.plot(contour[:, 1], contour[:, 0], linewidth=5, c=color_raw, zorder=1)
	if boob:
		for n, contour in enumerate(contours_inc):
			ax.plot(contour[:, 1], contour[:, 0], linewidth=1, c=color_inc, zorder=2)
	if booa and boob:
		#format current tip locations
		n_list,x_list,y_list = enumerate_tips(get_tips(contours_raw, contours_inc))
		if n_list:
		# if len(x_list)>0:
			ax.scatter(x=x_list, y=y_list, s=10, c='y', marker='*', zorder=3)
	return ax
