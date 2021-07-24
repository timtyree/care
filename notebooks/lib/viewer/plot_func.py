import matplotlib.pyplot as plt, numpy as np, pandas as pd
from ..utils import *
from ..measure import *
# general functions for plotting
# Tim Tyree
# 7.23.2021

def format_plot_general(**kwargs):
    return format_plot(**kwargs)

def format_plot(ax,xlabel,ylabel,fontsize=20,use_loglog=False,**kwargs):
    '''format plot formats the matplotlib axis instance, ax,
    performing routine formatting to the plot,
    labeling the x axis by the string, xlabel and
    labeling the y axis by the string, ylabel
    '''
    if use_loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(xlabel,fontsize=fontsize,**kwargs)
    ax.set_ylabel(ylabel,fontsize=fontsize,**kwargs)
    ax.tick_params(axis='both', which='major', labelsize=fontsize,**kwargs)
    ax.tick_params(axis='both', which='minor', labelsize=0,**kwargs)
    return True
