# my_initialization.py for worker
from . import *
import numpy as np, pandas as pd
#automate the boring stuff
from IPython import utils
import time, os, sys, re, shutil
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
    nb_dir = os.getcwd()
# %autocall 1
# %load_ext autoreload
# %autoreload 2
