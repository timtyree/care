# from spring import *
import numpy as np
from numba import njit

#compute spring force for a given edge configuration
@njit
def spring_force( vertex1, vertex2, d, d0, k0 ):
    '''returns the spring_force for vertex1 directed towards vertex2.  '''
    dhat = (vertex2 - vertex1)
    dhat /= np.linalg.norm(dhat)
    return - k0 * ( d - d0 ) * dhat

@njit
def compute_spring_forces(vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array):
    '''compute the net spring force at each vertex by iterating over the edges modeling springs.'''
    f0_array = 0. * vertex_array
    # f0_array = np.array(np.zeros_like(vertex_array, dtype=float))
    # dist_array  = np.zeros_like(vertex1_array, dtype=float)
    # force_array = np.zeros_like(vertex1_array, dtype=float)
    #     for i, (vertex1, vertex2, d, d0, k0) in enumerate(zip(vertex1_array, vertex2_array, d_array, d0_array, k0_array)):
    # dist_array[i] = vertex1 - vertex2
    imax = vertex1_array.shape[0]
    for i in range(imax):
        vertex1, vertex2, d, d0, k0 = vertex1_array[i], vertex2_array[i], d_array[i], d0_array[i], k0_array[i]
    # compute the spring force given edge in the local scope
        f0   = spring_force( vertex1, vertex2, d, d0, k0 )

        # add that spring force in the appropriate entries of f0_array
        f0_array[vid1_array[i]] = f0_array[vid1_array[i]] + f0
        f0_array[vid2_array[i]] = f0_array[vid2_array[i]] - f0
    return f0_array

##########################
# # Example Usage:
##########################
# import os, trimesh, numpy as np
# os.chdir(f'Data/spherical_meshes')
# mesh = trimesh.load('spherical_mesh_64.stl')
# os.chdir('../..')
# #subtract the center of mass
# mesh.vertices -= mesh.center_mass
# #normalize the mean radius to 1
# mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))

# #precompute edges, which are the force generating simplices here
# meu = mesh.edges_unique
# vid1_array, vid2_array = np.array(meu[:,0],dtype=int), np.array( meu[:,1],dtype=int)
# vertex1_array = np.array ( mesh.vertices[vid1_array]  , dtype=float) 
# vertex2_array = np.array ( mesh.vertices[vid2_array]  , dtype=float) 
# vertex_array = np.array(mesh.vertices, dtype=float) #only needed for shape

# #initialize spring force parameters
# d0_array = np.array(mesh.edges_unique_length, dtype=float)
# k0_array = 1 + np.zeros_like(vertex1_array[...,0], dtype=float)
# f0_array = np.array(np.zeros_like(mesh.vertices, dtype=float))

# #compute the current spring deformation states for each edge
# vertex1_array = np.array(mesh.vertices[vid1_array],dtype=float)
# vertex2_array = np.array( mesh.vertices[vid2_array],dtype=float)
# d_array = np.array(mesh.edges_unique_length, dtype=float)

# # compute the net spring force for each vertex
# f0_array = compute_spring_forces(vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)