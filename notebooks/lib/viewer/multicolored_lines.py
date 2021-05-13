#!/bin/bash env python3
#multicolored_lines.py
#Tim Tyree
#5.10.2021
# forked fromhttps://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from ..measure._utils_find_contours import split_contour_into_contiguous

def plotMulticoloredLine(fig,ax,x_values,y_values,c_values,cmap='coolwarm',use_colorbar=True,vmin=None,vmax=None,alpha=1.,lw=2):
    '''x_values,y_values,c_values are each 1-by-N numpy arrays.'''
    #define the relevant segments
    points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    if vmin is None:
        vmin=c_values.min()
    if vmax is None:
        vmax=c_values.max()
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
    # lc = LineCollection(segments, cmap='hot', norm=norm)
    # Set the values used for colormapping
    lc.set_array(c_values)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)
    if use_colorbar:
        fig.colorbar(line, ax=ax)

    # # Use a boundary norm instead
    # cmap = ListedColormap(['r', 'k', 'b'])
    # norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    # lc = LineCollection(segments, cmap=cmap, norm=norm)
    # lc.set_array(dydx)
    # lc.set_linewidth(2)
    # line = ax.add_collection(lc)
    # if use_colorbar:
    #     fig.colorbar(line, ax=ax)
    return None

def plotColoredContour(fig,ax,xy_values_lst,c_values_lst,
                      cmap='hot',use_colorbar=False,
                      vmin=0.,vmax=10.,lw=3,navg=20,alpha=1.):
    for i in range(len(c_values_lst)):
        c_values=np.abs(c_values_lst[i])#.copy()
        #compute moving average of c_values
        # for k in range(navg):
        #     c_values[1:]=(c_values[1:]+c_values[:-1])/2.
        c_lst=[]
        for j in range(c_values.shape[0]):
            c_lst.append(np.mean(c_values[j:j+navg]))
        c_values=np.array(c_lst)
        xy_values=xy_values_lst[i]
        contour_lst = split_contour_into_contiguous(xy_values)
        for contour in contour_lst:
            x_values=contour[:,0]
            y_values=contour[:,1]
            plotMulticoloredLine(fig,ax,
                               x_values,
                               y_values,
                               c_values,
                               cmap=cmap,
                               use_colorbar=use_colorbar,
                              vmin=vmin,vmax=vmax,lw=lw,alpha=alpha)
    return None

if __name__=='__main__':
    x = np.linspace(0, 3 * np.pi, 500)
    y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    x_values=np.sin(x)
    y_values=np.cos(x)
    c_values=dydx

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig, ax = plotMulticoloredLine(fig,ax,x_values,y_values,c_values,cmap='jet',use_colorbar=True)
#     ax.set_xlim([0,width])
#     ax.set_ylim([0,height])
    ax.set_xlim(x_values.min(), x_values.max())
    ax.set_ylim(y_values.min(), y_values.max())
    plt.show()
