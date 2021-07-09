import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def PlotMeanRadialVelocities(axs,
                             x_values,
                             y_values,
                             y_err_values,
                             y_hat_values,
                             alpha=0.5,
                             c='C2',
                             fontsize=18,
                             elinewidth=3,
                             markersize=4,
                             capsize=3,
                             xlim0=[0, 4],
                             xlim1=[0, 4],
                             **kwargs):

    x1_label=r'R (cm)'
    x2_label=r'1/R (cm$^{-1}$)'
    y_label=r'dR/dt (cm/ms)'

    ax = axs[0]
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
    ax.set_xlim(xlim0)
    # ax.set_ylim([-0.5,.1])
    ax.set_xlabel(x1_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=1)

    ax = axs[1]
    # ax.scatter(1/x_values,y_values,c=c,s=20,alpha=alpha)
    #plot error bars
    ax.errorbar(x=1 / x_values,
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
    ax.plot(1 / x_values, y_hat_values, c=c, lw=2, alpha=1, linestyle='solid')

    #format plot
    ax.set_xlim(xlim1)
    #     ax.set_ylim([-0.2, .2])
    ax.set_xlabel(x2_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=1)
    return True
