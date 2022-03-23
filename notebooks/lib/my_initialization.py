#my_initialization.py
# from .utils.operari import *
#automate the boring stuff
# from IPython import utils
import time, os, sys, re
import dask.bag as db
from inspect import getsource
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
    nb_dir = os.getcwd()

from numba import njit
import pandas as pd, numpy as np, matplotlib.pyplot as plt, json

# ignore user warnings
import warnings
warnings.simplefilter('ignore', UserWarning)

#load the libraries
from . import *

if not 'darkmode' in globals():
    darkmode=False
if darkmode:
	# For darkmode plots
	from jupyterthemes import jtplot
	jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

if not 'gpumode' in globals():
    gpumode=False
if gpumode:
    import cudf,pycuda
