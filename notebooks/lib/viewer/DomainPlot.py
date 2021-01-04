
from .. import *

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