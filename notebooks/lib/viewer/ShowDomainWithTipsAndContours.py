from .. import *
import matplotlib.pyplot as plt
def ShowDomain(img,dimgdt,x_values,y_values,c_values,V_threshold,t,inch=6,fontsize=16,vmin_img=0.,vmax_img=0.2,vmin_tips=0.,vmax_tips=1.,
                 area=25,frameno=1,save_fn=None,save_folder=None,save=False,annotating=False,axis=[0,200,0,200],cmap='Blues', **kwargs):
    #plot the system
    # figsize=(15,15); max_marker_size=800; lw=2;color_values = None
    #appears to work     contours1 = find_contours(img,    level = 0.5)
    # img_nxt=img+Delta_t*dimgdt
    n_tips = x_values.shape[0]
    contours1 = find_contours(img,        level = V_threshold)
    # contours2 = find_contours(img_nxt,    level = V_threshold)
    contours3 = find_contours(dimgdt,     level = 0.)

    fig, ax = plt.subplots(figsize=(inch,inch))
    ax.imshow(img, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    # ax.imshow(dimgdt, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    # ax.imshow(dimgdt*img, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    plot_contours_pbc(contours1, ax, linewidth=2, min_num_vertices=6, linestyle='--', alpha=0.5, color='red')#'blue')
    # plot_contours_pbc(contours2, ax, linewidth=2, min_num_vertices=6, linestyle='--', alpha=0.5, color='green')
    plot_contours_pbc(contours3, ax, linewidth=2, min_num_vertices=6, linestyle='-', alpha=0.5, color='orange')

    #plot spiral tips. color inner spiral tip by slow variable
    ax.scatter(x=x_values, y=y_values, s=270, c=1+0.*c_values, marker='*', zorder=3, alpha=1., vmin=vmin_tips,vmax=vmax_tips)
    ax.scatter(x=x_values, y=y_values, s=135, c=c_values, marker='*', zorder=3, alpha=1., vmin=vmin_tips,vmax=vmax_tips, cmap=cmap)
    # ax.scatter(x=x_values, y=y_values, s=270, c='yellow', marker='*', zorder=3, alpha=1.)
    # ax.scatter(x=x_values, y=y_values, s=45, c='blue', marker='*', zorder=3, alpha=1.)
    if annotating:
        time_step_string=f"\tt = {t:.3f} ms"
        message_string=f"\tnum. = {n_tips}"
        # x=.5#0.
        # y=.95
        # horizontalalignment='left'#'left'
        # ax.text(x,y,time_step_string,
        #         horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
        # x=.5#0.
        # y=.9
        # horizontalalignment='left'#'left'
        # ax.text(x,y,message_string,
        #         horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
        # x=.5
        # y=.01
        # horizontalalignment='center'
        # ax.text(x,y,f"Area = {area} cm^2, V. Threshold = {V_threshold}",
        #         horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
        # ax.set_title(f"Area = {area} cm^2, V. Threshold = {V_threshold}, Num. Tips = {n_tips}\n", color='white', loc='left',pad=0)


        ax.text(.0,.95,time_step_string,
                horizontalalignment='left',color='white',fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(.0,.9,message_string,
                horizontalalignment='left',color='white',fontsize=fontsize,
                transform=ax.transAxes)

    ax.axis(axis)
    #     ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    if not save:
        plt.show()
        return fig
    else:
        if save_fn is None:
            save_fn = f"img{frameno:07d}.png"
            frameno += 1
    #         plt.tight_layout()
        if save_folder is not None:
            os.chdir(save_folder)
        plt.savefig(save_fn,dpi=720/inch, bbox_inches='tight',pad_inches=0);
        plt.close();
#         print ( save_fn )
#         return frameno
    return fig

#alias
show_buffer_w_tips_and_contours=ShowDomain

def PlotMyDomain(img,dimgdt,Delta_t,x_values,y_values,c_values,V_threshold,t,inch=6,fontsize=16,vmin_img=0.,vmax_img=0.2,
                 area=25,frameno=1,save_fn=None,save_folder=None,save=False,annotating=False,axis=[0,200,0,200], **kwargs):
    #plot the system
    # figsize=(15,15); max_marker_size=800; lw=2;color_values = None
    #appears to work     contours1 = find_contours(img,    level = 0.5)
    img_nxt=img+Delta_t*dimgdt
    n_tips = x_values.shape[0]
    contours1 = find_contours(img,        level = V_threshold)
    contours2 = find_contours(img_nxt,    level = V_threshold)
    contours3 = find_contours(dimgdt,     level = 0.)

    fig, ax = plt.subplots(figsize=(inch,inch))
    # ax.imshow(img, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    ax.imshow(dimgdt, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    # ax.imshow(dimgdt*img, cmap=plt.cm.gray,vmin=vmin_img,vmax=vmax_img)
    plot_contours_pbc(contours1, ax, linewidth=2, min_num_vertices=6, linestyle='--', alpha=0.5, color='red')#'blue')
    plot_contours_pbc(contours2, ax, linewidth=2, min_num_vertices=6, linestyle='--', alpha=0.5, color='green')
    plot_contours_pbc(contours3, ax, linewidth=2, min_num_vertices=6, linestyle='-', alpha=0.5, color='orange')

    #plot spiral tips. color inner spiral tip by slow variable
    ax.scatter(x=x_values, y=y_values, s=270, c=1+0.*c_values, marker='*', zorder=3, alpha=1., vmin=0,vmax=1)
    ax.scatter(x=x_values, y=y_values, s=135, c=c_values, marker='*', zorder=3, alpha=1., vmin=0,vmax=1, cmap='Blues')
    # ax.scatter(x=x_values, y=y_values, s=270, c='yellow', marker='*', zorder=3, alpha=1.)
    # ax.scatter(x=x_values, y=y_values, s=45, c='blue', marker='*', zorder=3, alpha=1.)
    if annotating:
        x=.5#0.
        y=.95
        horizontalalignment='center'#'left'
        ax.text(x,y,f"Current Time = {t:.3f} ms",
                horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
        x=.5#0.
        y=.9
        horizontalalignment='center'#'left'
        ax.text(x,y,f"Num. of Tips  = {n_tips}",
                horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
        x=.5
        y=.01
        horizontalalignment='center'
        ax.text(x,y,f"Area = {area} cm^2, V. Threshold = {V_threshold}",
                horizontalalignment=horizontalalignment,color='white',fontsize=fontsize,transform=ax.transAxes)
    # ax.set_title(f"Area = {area} cm^2, V. Threshold = {V_threshold}, Num. Tips = {n_tips}\n", color='white', loc='left',pad=0)
    ax.axis(axis)
    #     ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    if not save:
        plt.show()
        return fig
    else:
        if save_fn is None:
            save_fn = f"img{frameno:07d}.png"
            frameno += 1
    #         plt.tight_layout()
        if save_folder is not None:
            os.chdir(save_folder)
        plt.savefig(save_fn,dpi=720/inch, bbox_inches='tight',pad_inches=0);
#         plt.close();
#         print ( save_fn )
#         return frameno
    return fig
