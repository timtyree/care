#!/usr/bin/env python3
# # bluf.py generates rows*columns graphs per page and each graph contains num_samples data points.
# # this module was artesianally forked from a stackoverflow post found here:
# # some care was required of me to add new lines, debug, and make this functionality into a module.
# # feel free to donate at: <TODO: add url for my paypal>
# # Programmer: Tim Tyree
# # Date: 7.23.2021
# from lib.viewer import *
from .DataPlotterClass import *
from .pdf_utils import *
import matplotlib.backends.backend_pdf, matplotlib.pyplot as plt,random, os
# from ...lib import *
# from core.viewer import *

def plotter(ax, plotter_function, input_arg):
    '''input_arg can be any input argument,
    such as a string directing the interpreter
    to an absolute file path containing data
    to be plotted
    a plotter_function that takes an axis, ax, and a string, input_fn, and plots it with matplotlib
    '''
    plotter_function(ax, input_arg)
    return True

def gener_bluf(task_lst,
               bluf_dir,
               figsize=(8.5, 11),
               rows=3,
               cols=2,
               bbox_inches='tight',
               save_tight=True):
    '''generates the bottom line up front, saving as a pdf to the absolute file path, bluf_dir, which is a string instance'''
    pdf = DataPlotter(pdfpath=bluf_dir,
                      rows=rows,
                      cols=cols,
                      bbox_inches=bbox_inches,
                      save_tight=save_tight)
    for j in range(len(task_lst)):
        plotter_function, arg = task_lst[j]
        ax = pdf.get_next_plot()
        w, h = figsize
        set_size(w, h, ax)
        #plot the next data
        plotter(ax, plotter_function, arg)
    #saves pdf
    pdf.close()
    return True


if __name__=='__main__':
    # settings
    nb_dir=os.path.dirname(os.path.dirname(os.getcwd()))
    bluf_dir = f"{nb_dir}/Figures/bluf.pdf"
    print(f'saving .pdf to\n{bluf_dir}')

    #define an example plotter_function
    def my_plotter_function(ax, data):
        '''a plotter_function as an example'''
        x_values, y_values, c_values = np.array(data).T
        c_values = np.abs(c_values)
        s = 25
        alpha = .4
        xlabel = 'xlabel'
        ylabel = 'ylabel'
        vmin = 0#np.min(c_values)
        vmax = 800#np.max(c_values)
        ax.scatter(x=-1*x_values,
                   y=y_values-x_values**2,
                   c=c_values*np.abs(x_values),
                   alpha=alpha,
                   s=s*np.sqrt(np.abs(x_values)),
                   vmin=vmin,
                   vmax=vmax,
                   cmap='hsv')
        format_plot(ax, xlabel, ylabel)
        return True

    # get some  data for a randomly generated rainbow
    data = []
    for i in range(0, 50):
        data.append([i, float(i + random.randrange(-50, 50)) / 100, 5])
    dataA=np.array(data)
    dataB=dataA.copy()
    dataB[:,0]=-1*dataB[:,0]#-40
    dataC=dataA.copy()
    # dataC[:,-1]=2*dataC[:,-1]
    dataD=dataB.copy()
    # dataD[:,-1]=2*dataD[:,-1]

    #define the lists of plotter tasks
    task_lst = [
        (my_plotter_function, dataA),
        (my_plotter_function, dataB),
        (my_plotter_function, dataC),
        (my_plotter_function, dataD),
    ]

    gener_bluf(task_lst, bluf_dir)

    #open the outputed .pdf automatically
    import webbrowser
    webbrowser.open_new(r'file://' + bluf_dir)

#(deprecated) # https://community.esri.com/t5/python-documents/creating-multiple-graphs-per-page-using-matplotlib/ta-p/916434
# pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
# cnt = 0
# # figs = plt.figure()
# for data_chunk in chunks(data, 600):
#     plot_num = 321
#     fig = plt.figure(figsize=(10, 10)) # inches
#     for sub_chunk in chunks(data_chunk, 100):
#         cnt += 1
#         d = [a[0] for a in sub_chunk]
#         z = [a[1] for a in sub_chunk]
#         zv = [a[2] for a in sub_chunk]
#         print ( plot_num )
#         plt.subplot(plot_num)
#         # plot profile, define styles
#         plt.plot(d,z,'r',linewidth=0.75)
#         plt.plot(d,z,'ro',alpha=0.3, markersize=3)
#         plt.plot(d,zv,'k--',linewidth=0.5)
#         plt.xlabel('Distance from start')
#         plt.ylabel('Elevation')
#         plt.title(f'Profile {cnt} using Python matplotlib')
#         # change font size
#         plt.rcParams.update({'font.size': 8})
#         plot_num += 1
#         pdf.savefig(fig)
#         pdf.close()
