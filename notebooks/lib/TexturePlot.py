import numpy as np
import matplotlib.pyplot as plt
from lib.get_tips import *
from lib.intersection import *

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
