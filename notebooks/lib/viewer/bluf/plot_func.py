import matplotlib.pyplot as plt, numpy as np, pandas as pd
# general functions for plotting
# Tim Tyree
# 7.23.2021

def format_plot(ax=None,xlabel=None,ylabel=None,fontsize=20,use_loglog=False,xlim=None,ylim=None,use_bigticks=True,**kwargs):
    '''format plot formats the matplotlib axis instance, ax,
    performing routine formatting to the plot,
    labeling the x axis by the string, xlabel and
    labeling the y axis by the string, ylabel
    '''
    if not ax:
        ax=plt.gca()
    if use_loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if xlabel:
        ax.set_xlabel(xlabel,fontsize=fontsize,**kwargs)
    if ylabel:
        ax.set_ylabel(ylabel,fontsize=fontsize,**kwargs)
    if use_bigticks:
        ax.tick_params(axis='both', which='major', labelsize=fontsize,**kwargs)
        ax.tick_params(axis='both', which='minor', labelsize=0,**kwargs)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_xlim(ylim)
    return True
