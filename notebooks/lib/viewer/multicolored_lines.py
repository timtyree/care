#!/bin/bash env python3
#multicolored_lines.py
#Tim Tyree
#5.10.2021
# forked fromhttps://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def plotMulticoloredLine(fig,ax,x_values,y_values,c_values,cmap='jet',use_colorbar=True,vmin=None,vmax=None):
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
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # lc = LineCollection(segments, cmap='hot', norm=norm)
    # Set the values used for colormapping
    lc.set_array(c_values)
    lc.set_linewidth(2)
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
    return fig,ax

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