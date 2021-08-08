import scipy.stats as stats,numpy as np, pandas as pd, matplotlib.pyplot as plt
def plot_hist_with_kde(data, ax, bandwidth = 0.3):
    #set number of bins using Freedman and Diaconis
    q1 = np.percentile(data,25)
    q3 = np.percentile(data,75)

    n = len(data)**(.1/.3)
    rng = max(data) - min(data)
    iqr = 2*(q3-q1)
    bins = int((n*rng)/iqr)

    x = np.linspace(min(data),max(data),200)

    kde = stats.gaussian_kde(data)
    kde.covariance_factor = lambda : bandwidth
    kde._compute_covariance()

    ax.plot(x,kde(x),'r') # distribution function
    ax.hist(data,bins=bins,density=True) # histogram
