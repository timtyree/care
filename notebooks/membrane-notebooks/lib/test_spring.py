import os, trimesh, numpy as np, sys


sys.path.append("..") 
from lib import *
# sys.path.append("lib") 
sys.path.append("../..") 
from lib import *

# if not 'nb_dir' in globals():
#     nb_dir = os.getcwd()
# sys.path.append("../lib") 
# from lib import *
# sys.path.append("lib") 
# from lib import *
# from spring import *

def test_spring_force_computation(mesh):

    #precompute edges, which are the force generating simplices here
    meu = mesh.edges_unique
    vid1_array, vid2_array = np.array(meu[:,0],dtype=int), np.array( meu[:,1],dtype=int)
    vertex1_array = np.array ( mesh.vertices[vid1_array]  , dtype=float) 
    vertex2_array = np.array ( mesh.vertices[vid2_array]  , dtype=float) 
    vertex_array = np.array(mesh.vertices, dtype=float) #only needed for shape

    #initialize spring force parameters
    d0_array = np.array(mesh.edges_unique_length, dtype=float)
    k0_array = 1 + np.zeros_like(vertex1_array[...,0], dtype=float)
    f0_array = np.array(np.zeros_like(mesh.vertices, dtype=float))

    #compute the current spring deformation states for each edge
    vertex1_array = np.array(mesh.vertices[vid1_array],dtype=float)
    vertex2_array = np.array( mesh.vertices[vid2_array],dtype=float)
    d_array = np.array(mesh.edges_unique_length, dtype=float)

    assert ((mesh.edges_unique[mesh.edges_unique_inverse] == mesh.edges_sorted).all())

    def length(vertex1, vertex2):
        return np.linalg.norm( vertex1 - vertex2)

    #test mesh.edges_unique_length gives the correct edge length 
    assert ( length(vertex1_array[0],vertex2_array[0]) == mesh.edges_unique_length[0] )
    assert ( length(vertex1_array[1],vertex2_array[1]) == mesh.edges_unique_length[1] )
    assert ( length(vertex1_array[2],vertex2_array[2]) == mesh.edges_unique_length[2] )
    
    #test type consistency as a proxy test that the types are not tracked-array instances
    # assert ( type(dist_array) == type (d0_array) )
    assert ( type(f0_array) == type (k0_array) )
    assert ( type(vertex1_array) == type (d0_array) )
    assert ( type(vertex2_array) == type (d0_array) )
    assert ( type(vid1_array) == type (d0_array) )
    assert ( type(vid2_array) == type (d0_array) )

    # test that spring_force pushes vertex1 in the direction of vertex2 when the spring is twice the equilibrium length, d0
    vertex1, vertex2, d, d0, k0  = (np.array([ 0.71367072, -0.45411208,  0.53182509]),np.array([ 0.54048159, -0.63460453,  0.61003465]),0.26208535311672354,0.26208535311672354,1.0)
    dhat = (vertex1 - vertex2)
    dhat/= np.linalg.norm(dhat)

    # test that an extended spring is attractive in the edge direction
    f0   = spring_force( vertex1, vertex2, d, d0/2, k0 )
    assert ( np.isclose ( f0.dot(dhat)/np.linalg.norm(f0) , 1 ) ) 
    # test that a compressed spring is repulsive in the edge direction
    f0   = spring_force( vertex1, vertex2, d, d0*2, k0 )
    assert ( np.isclose ( f0.dot(dhat)/np.linalg.norm(f0) , -1 ) ) 
    
    # compute the net spring force for each vertex
    # f0_array = compute_spring_forces(vertex1_array, vertex2_array, d_array, d0_array, k0_array)
    f0_array = compute_spring_forces(vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)
    assert (np.isclose( np.mean(f0_array) , 0 ) )  #test average spring force is zero
    assert ( np.isclose( np.var(f0_array) , 0 ) )  #test variance of spring force at least 0.02
    return True

if __name__=='__main__':
    import os, trimesh, numpy as np
    from spring import *

    os.chdir(f'../Data/spherical_meshes')
    mesh = trimesh.load('spherical_mesh_64.stl')
    os.chdir('../../lib')
    #subtract the center of mass
    mesh.vertices -= mesh.center_mass
    #normalize the mean radius to 1
    mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))
    test_spring_force_computation(mesh)
    print('ignore ^that line. \n test_spring_force_computation passed.  Yay!')

