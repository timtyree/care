# Module for mechanical model
from lib.geom_func import *
import numpy as np

#Example Usage
# #compute P, first Piola-Kirchoff stress tensor
# mu  = 1. #first  lame parameter, pressure_units
# lam = 1. #second lame parameter, pressure_units
# delta = .01 #small thickness of membrane, length_units
# one = np.eye(3)
# calc_P = get_calc_P(mu, lam, one, delta)

def preprocessing():
    pass

def compute_elastic_forces():
    pass

def compute_force_differentials():
    pass

# pycuda.gpuarray.GPUArray(shape, dtype, *, allocator=None, order='C')

######################################
# Nodal Elastic Forces
######################################
#TODO: njit this! 
def get_calc_P(mu, lam, one, delta):
    '''returns the first Piola-Kirchoff stress tensor ( times the constant membrane thickness, delta) 
    from the corotated linear constitutive model for elastic stress.
    Example Usage - given deformation matrix F = S.dot(R):
    mu = 1.; lam = 1.; delta = 0.1; one = np.eye(3);
    calc_P = get_calc_P(mu, lam, one, delta)
    P = calc_P(S, R)'''
    return lambda S, R: delta * (2*mu*(S - one).dot(R) + lam*np.trace(S - one) * R)

def calc_outward_normals(trim):
    '''
    returns the outward unit normal vectors for the triangle, trim.
    trim is a 3x3 numpy array.
    '''
    #compute local unit vectors of triangle's shape and center of mass (com)
    d1, d2, A = get_shape(trim)
    c   = trim[2] - trim[1]
    c  /= np.linalg.norm(c)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    A  /= np.linalg.norm(A)
    com     = (trim[0]+trim[1]+trim[2])/3

    #precompute the outward normals in material space
    #and test that the outward normals are indeed outward
    N1 = np.cross(d1,A)
    N1tilde = (trim[0]+trim[1])/2 - com
    if N1.dot(N1tilde)<0:
        N1 *= -1
        print("N1 was flipped.")
    N2 = np.cross(-d2,A)
    N2tilde = (trim[0]+trim[2])/2 - com
    if N2.dot(N2tilde)<0:
        N2 *= -1
        print("N2 was flipped.")
    N3 = np.cross(c,A)
    N3tilde = (trim[2]+trim[1])/2 - com
    if N3.dot(N3tilde)<0:
        N3 *= -1
        print("N3 was flipped.")
    return N1,N2,N3

#TODO: njit this!!
def calc_elastic_force(trim, tris, calc_P, N1N2N3):
    '''N1N2N3 are the precomputed outward normals of trim returned by compute_outward_normals
    calc_P is returned by get_calc_P,
    tris and trim are 3x3 numpy arrays of floats'''
    S, R = get_SR(trim,tris)
    P    = calc_P(S, R)
    N1,N2,N3 = N1N2N3  #TODO: precompute these if they're too slow

    #compute the net force on each side of tris
    D1 = np.linalg.norm(tris[1]-tris[0])
    D2 = np.linalg.norm(tris[2]-tris[0])
    C  = np.linalg.norm(tris[2]-tris[1])
    F1 = - P.dot(N1) * D1
    F2 = - P.dot(N2) * D2
    F3 = - P.dot(N3) * C

    #compute the equivalent nodal forces on each vertex of tris
    nf0 = (F1 + F2)/2.  #the nodal force on the vertex at tris[0]
    nf1 = (F1 + F3)/2.  #the nodal force on the vertex at tris[1]
    nf2 = (F3 + F2)/2.  #the nodal force on the vertex at tris[2]
    return nf0, nf1, nf2

#TODO: njit this! 
# this takes 0.5 seconds to compute the net forces on 500 rows.  Not ideal
# a number of functions need to be njit'd to get this down to ~30ms like pressure
def calc_elastic_forces(net_elastic_forces, mesh, comp, outward_normals):
    #for each face
    for fid, face in enumerate(mesh.faces):
        vid0,vid1,vid2 = mesh.faces[fid]
        trim = np.array(mesh.triangles[fid])
        tris = np.array(comp.triangles[fid])
        #compute elastic nodal forces
        N1N2N3 = outward_normals[fid]
        nf0, nf1, nf2 = calc_elastic_force(trim, tris, calc_P, N1N2N3)
        net_elastic_forces[vid0] += nf0
        net_elastic_forces[vid1] += nf1
        net_elastic_forces[vid2] += nf2
