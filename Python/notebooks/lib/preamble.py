#preamble for dev ipynb

#pylab
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pylab import imshow, show

#use cuda via numba
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8

#automate the boring stuff
from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
    nb_dir = os.getcwd()

#%autocall 1
%load_ext autoreload
%autoreload 2


#nota bene: you can increase the ram allocated to the virtual machine running jupyter with 
#$ jupyter notebook --NotebookApp.max_buffer_size=your_value
