import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def PlotMeanSquaredDisplacements(ax,
                             x_values,
                             y_values,
                             y_err_values,
                             y_hat_values,
                             alpha=0.5,
                             c='C3',
                             fontsize=18,
                             elinewidth=3,
                             markersize=4,
                             capsize=3,
                             xlim=[0, 0.5],
                             ylim=[0, 500],
                             **kwargs):

    x_label=r'lag (ms)'
    y_label=r'MSD (cm$^2$)'

    # ax.scatter(x_values,y_values,c=c,s=20,alpha=alpha)
    #plot error bars
    ax.errorbar(x=x_values,
                y=y_values,
                yerr=y_err_values,
                c=c,
                alpha=alpha,
                fmt='o',
                markersize=markersize,
                ecolor=c,
                elinewidth=elinewidth,
                errorevery=1,
                capsize=capsize)
    #plot fits
    ax.plot(x_values, y_hat_values, c=c, lw=2, alpha=1, linestyle='solid')

    #format plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=1)
    return True
