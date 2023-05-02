#generates and saves a scatter plot as png
#Tim Tyree
#6.8.2021
import numpy as np, matplotlib.pyplot as plt,os

def ScatterPlotSnapshot(x_values,y_values,t,width,height,marker='*',c='gray',s=200,alpha=0.8,
               fontsize=18,inch=6,textcolor='k',message='',annotating=True):
    '''performs a simple scatter plot and returns a figure that may be saved as png.
    Example Usage:
annotating=True
message='initial positions'
fig=ScatterPlotSnapshot(x_values,y_values,t,width=L,height=L,
                        annotating=annotating,message=message,inch=4)
plt.show()
    '''
    figsize=(inch,inch)
    fig,ax=plt.subplots(1,figsize=figsize)
    ax.scatter(x_values,y_values,marker=marker,c=c,s=s,alpha=alpha)
    ax.axis('off')
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    if annotating:
        time_step_string=f"  t = {t:.2f} s"
        message_string=f"  num. = {x_values.shape[0]}"
        ax.text(.5,.95,message,
                horizontalalignment='center',color=textcolor,fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(.97,.05,time_step_string,
                horizontalalignment='right',color=textcolor,fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(.97,.0,message_string,
                horizontalalignment='right',color=textcolor,fontsize=fontsize,
                transform=ax.transAxes)
    return fig

def SaveScatterPlotSnapshot(x_values,y_values,t,width,height,frameno,save_folder,
                            annotating=True,message='',
                            saving = True, inch = 6, save_fn=None, **kwargs):
    '''generates and saves ScatterPlotSnapshot
    Example Usage:
save_folder=f"{nb_dir}/Figures/mov"
frameno=1;save_fn=None;inch=6;message='without forces';annotating=True
SaveScatterPlotSnapshot(x_values,y_values,t,width=L,height=L,frameno=frameno,save_folder=save_folder
                            annotating=annotating,message=message)
frameno+=1
    '''
    fig=ScatterPlotSnapshot(x_values,y_values,t,width,height,
                        annotating=annotating,message=message)
    if not saving:
        plt.show()
    else:
        os.chdir(save_folder)
        if save_fn is None:
            save_fn = f"img{frameno:07d}.png"
        plt.savefig(save_fn,dpi=720/inch, bbox_inches='tight',pad_inches=0,
                    facecolor='white');
        plt.close()
