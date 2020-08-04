#initialization for viewer.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

#automate the boring stuff
from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():     nb_dir = os.getcwd()
sys.path.append("../lib") 
from lib import *
sys.path.append("lib") 
from lib import *

# the visualization tools involved here for triangular meshes is
import trimesh
import pyglet
from numba import njit
# from numba import cuda

def read_mesh(file_name):
    mesh = trimesh.load(file_name);
    return mesh

def center_mesh(mesh, V_initial=None):
    #subtract the center of mass
    mesh.vertices -= mesh.center_mass
    #normalize the mean radius to 1
    # V_initial = 26 #cm^3
    if V_initial is not None:
        # mesh.vertices /= np.cbrt((mesh.volume)*3/(4*np.pi))
        mesh.vertices *= np.cbrt(V_initial/mesh.volume)
        #normalize the mean radius to R
        # R = 1. #unit length
        # mesh.vertices *= R/np.mean(np.linalg.norm(mesh.vertices,axis=1))
        return mesh
    else:
        return mesh

def write_mesh(mesh,file_name):
    '''writes mesh to a file_name ending in stl'''
    mesh.export(file_name, file_type='stl');
    return mesh    