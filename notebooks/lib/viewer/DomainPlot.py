from .. import *
from ..viewer.multicolored_lines import plotColoredContour

def ShowTipsAndColoredContours(fig,ax,dict_tips,
						   fontsize=18,cmap='hot',
						  annotating=True,textcolor='white',
						  vmin_tips=-np.pi/4.,vmax_tips=np.pi/4.):
	xy_values_lst=dict_tips['greater_xy_values']
	c_values_lst =dict_tips['greater_curvature_values']
	plotColoredContour(fig,ax,xy_values_lst,c_values_lst,
						  cmap='hot',use_colorbar=False,
					   vmin=0.,vmax=3.,lw=3,navg=60,alpha=0.05)
	#plot the list of lesser contours
	xy_values_lst=dict_tips['lesser_xy_values']
	c_values_lst =dict_tips['lesser_curvature_values']
	plotColoredContour(fig,ax,xy_values_lst,c_values_lst,
						  cmap='hot',use_colorbar=False,
					   vmin=0.,vmax=3.,lw=3,navg=60,alpha=0.5)


	#plot spiral tips. color inner spiral tip by slow variable
	x_values=np.array(dict_tips['x'])
	y_values=np.array(dict_tips['y'])
	c_values=np.array(dict_tips['phi'])
	n_tips = x_values.shape[0]
	ax.scatter(x=x_values, y=y_values, s=300, c=1+0.*c_values, marker='*', zorder=3, alpha=.5, vmin=0.,vmax=1.)
	ax.scatter(x=x_values, y=y_values, s=100, c=c_values, marker='*',
			   zorder=3, alpha=.8, vmin=vmin_tips,vmax=vmax_tips, cmap='bwr')

	if annotating:
		t=dict_tips['t']
		time_step_string=f"  t = {t:.1f} ms"
		message_string=f"  num. = {n_tips}"
		ax.text(.97,.13,time_step_string,
				horizontalalignment='right',color=textcolor,fontsize=fontsize,
				transform=ax.transAxes)
		ax.text(.97,.05,message_string,
				horizontalalignment='right',color=textcolor,fontsize=fontsize,
				transform=ax.transAxes)
	return None

#plot system with colored curvature
def SaveTipsAndColoredContours(img,frameno,dict_tips,save_folder=None,save_fn=None,vmin_img=-85.,vmax_img=35.,inch=5):
	save=True
	figsize=(inch,inch)
	fig, ax = plt.subplots(figsize=figsize)#, sharex=True, sharey=True)
	ax.imshow(img,vmin=vmin_img,vmax=vmax_img,cmap='gray')
	ShowTipsAndColoredContours(fig,ax,dict_tips,
							   fontsize=18,cmap='hot',
							  annotating=True,textcolor='white',
							  vmin_tips=-np.pi/4.,vmax_tips=np.pi/4.)
	width,height=img.shape[:2]
	ax.set_xlim([0,width])
	ax.set_ylim([0,height])
	ax.axis('off')

	if not save:
		plt.show()
	else:
		if save_fn is None:
			save_fn = f"img{frameno:07d}.png"
			frameno += 1
	#         plt.tight_layout()
		if save_folder is not None:
			os.chdir(save_folder)
		plt.savefig(save_fn,dpi=720/inch, bbox_inches='tight',pad_inches=0);
		plt.close();
	return frameno


#TODO(later): plot these tips with their u's and v's
# Display the image and plot all pbc contours found properly!
def save_plot_as_png(img, dimgdt, x_values, y_values, c_values, n_tips, t, save_folder, frameno,
	save = True, inch = 6, save_fn=None, **kwargs):
	''''''
	fig, ax = plt.subplots(figsize=(inch,inch))

	#appears to work     contours1 = find_contours(img,    level = 0.5)
	contours1 = find_contours(img,    level = level1)
	contours2 = find_contours(dimgdt, level = level2)

	# ax.imshow(img, cmap=plt.cm.gray)
	ax.imshow(dimgdt, cmap=plt.cm.gray)
	# ax.imshow(dimgdt*img, cmap=plt.cm.gray)


	plot_contours_pbc(contours1, ax, linewidth=2, min_num_vertices=6, linestyle='-', alpha=0.5, color='blue')
	plot_contours_pbc(contours2, ax, linewidth=2, min_num_vertices=6, linestyle='--', alpha=0.5, color='orange')

	#plot spiral tips. color inner spiral tip by slow variable
	ax.scatter(x=x_values, y=y_values, s=270, c=1+0.*c_values, marker='*', zorder=3, alpha=1., vmin=0,vmax=1)
	ax.scatter(x=x_values, y=y_values, s=45, c=c_values, marker='*', zorder=3, alpha=1., vmin=0,vmax=1, cmap='Blues')
	# ax.scatter(x=x_values, y=y_values, s=270, c='yellow', marker='*', zorder=3, alpha=1.)
	# ax.scatter(x=x_values, y=y_values, s=45, c='blue', marker='*', zorder=3, alpha=1.)

	ax.text(.0,.95,f"Current Time = {t:.1} ms",
			horizontalalignment='left',color='white',fontsize=16,
			transform=ax.transAxes)
	ax.text(.0,.9,f"Num. of Tips  = {n_tips}",
			horizontalalignment='left',color='white',fontsize=16,
			transform=ax.transAxes)
	ax.text(.5,.01,f"Area = {25}cm^2, V. Threshold = {level1}",
			horizontalalignment='center',color='white',fontsize=16,
			transform=ax.transAxes)

	# ax.set_title(f"Area ={25}cm^2, V. Threshold ={V_threshold}, Num. Tips ={n_tips}", color='blue', loc='left',pad=0)
	ax.axis([0,200,0,200])
#     ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])

	if not save:
		plt.show()
		return fig
	else:
		os.chdir(save_folder)
		if save_fn is None:
			save_fn = f"img{frameno:07d}.png"
			frameno += 1
#         plt.tight_layout()
		plt.savefig(save_fn,dpi=720/inch, bbox_inches='tight',pad_inches=0);
		plt.close();
		#     print(f'figure saved in {save_fn}.')
		#     plt.savefig('example_parameterless_tip_detection_t_600.png')
		return frameno
