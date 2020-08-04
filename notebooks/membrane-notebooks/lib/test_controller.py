# test_controller.py
import os, trimesh, numpy as np, sys
from controller import *
from spring import *
from .vertex_shader import *
# if not 'nb_dir' in globals():
# 	nb_dir = os.getcwd()
# sys.path.append("..") 
# from lib import *
# # sys.path.append("lib") 
# sys.path.append("../..") 
# from lib import *
# from spring import *
# from controller import *



def test_spring_exists():
    # test spring_exists for a simple case
    assert ( spring_exists(np.array([3,4,1]), np.array([1,2,3]), vid1=3,vid2=1)     )
    assert ( not spring_exists(np.array([3,4,1]), np.array([1,2,3]), vid1=7,vid2=7) )
    assert ( not spring_exists(np.array([3,4,1]), np.array([1,2,3]), vid1=3,vid2=3) )
    assert ( not spring_exists(np.array([3,4,1]), np.array([1,2,3]), vid1=1,vid2=1) )
    return True

def test_spring_add_remove(Omegat):    
    #tests for adding/removing springs
    vid1_array = Omegat.precomputed_arguments['vid1_array']
    vid2_array = Omegat.precomputed_arguments['vid2_array']

    blacklist = [(7,7)]
    whitelist = [(7,7)]
    assert ( not spring_exists(vid1_array, vid2_array, vid1=7,vid2=7) ) 

    #test that add_springs and then remove_springs does nothing when told to do nothing
    len_original = 756
    assert ( len(vid1_array) == len_original )
    assert ( len(vid2_array) == len_original )

    v1, v2 = remove_springs(vid1_array, vid2_array, blacklist=[]) 
    assert ( len(v1) == len_original )
    assert ( len(v2) == len_original )

    v1, v2 = remove_springs(vid1_array, vid2_array, blacklist) 
    assert ( len(v1) == len_original )
    assert ( len(v2) == len_original )

    v1, v2 = add_springs(vid1_array, vid2_array, whitelist=[]) 
    assert ( len(v1) == len_original )
    assert ( len(v2) == len_original )

    v1, v2 = add_springs(v1, v2, whitelist) 
    assert ( len(v1) == len_original +1)
    assert ( len(v2) == len_original +1)
    assert ( spring_exists(v1, v2, vid1=7,vid2=7) )

    #add_springs does not add a spring that already exists
    v1, v2 = add_springs(v1, v2, whitelist) 
    assert ( len(v1) == len_original +1)
    assert ( len(v2) == len_original +1)
    assert ( spring_exists(v1, v2, vid1=7,vid2=7) )

    #remove_springs removes a spring that exists
    v1, v2 = remove_springs(v1, v2, whitelist) 
    assert ( len(v1) == len_original )
    assert ( len(v2) == len_original )
    assert ( not spring_exists(v1, v2, vid1=7,vid2=7) )
    return True

def test_initialize_system(mesh, mass_array):
    #test precompute_mesh
    Omegat = precompute_mesh(mesh, mass_array)

    # test precompute_mesh preserved the number of vertices/faces
    assert ( mesh.vertices.shape == (254, 3) )
    assert ( mesh.faces.shape    == (504, 3) )
    assert ( Omegat.vertices.shape == (254, 3) )
    assert ( Omegat.faces.shape    == (504, 3) )
    # test precompute_mesh created the right properties
    assert ( Omegat.precomputed_arguments is not None ) 
    assert ( Omegat.Omega0 is not None ) 

    #test compute_mesh_update
    # test that zero displacement updates Omegat to itself
    u = np.zeros_like(Omegat.X, dtype=float)
    retval = compute_mesh_update(Omegat,u)
    assert ( np.isclose(retval.vertices, Omegat.vertices).all() )

    # test that updating retval does updates Omegat.  
    a = Omegat.vertices[0][0] #first coord of first vertex
    retval.vertices += 1
    b = Omegat.vertices[0][0]
    assert ( a != b )


    #test initialize_system
    Omegat = initialize_system(mesh, mass_array)
    ud  = np.zeros_like(Omegat.X, dtype=float)
    assert ( type(Omegat.u) is type(ud) ) 
    assert ( type(Omegat.ud) is type(ud) ) 
    assert ( type(Omegat.udd) is type(ud) ) 
    assert ( type(Omegat.X) is type(ud) ) 
    return True

def test_simplest_controller_at_equilibrium():
    #test getting use a constant time step size, h
    h = get_h(h=0.01)#, beta=1.0, acceptedQ=True)
    assert(h==0.01)

    #initialize system at equilibrium
    import os, trimesh, numpy as np
    from lib.spring import *
    os.chdir(f'Data/spherical_meshes')
    mesh = trimesh.load('spherical_mesh_64.stl')
    os.chdir('../..')
    #subtract the center of mass
    mesh.vertices -= mesh.center_mass
    #normalize the mean radius to 1
    mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))
    #compute vertex masses for the displacement invariant barycentric discretization of a 2-D surface.
    mass_array = np.array ( [get_mass(vid, mesh, density = 1.) for vid in range(mesh.vertices.shape[0])] )
    Omegat = initialize_system(mesh, mass_array)
    tme = 0.

    #test getting acceleration due to spring forces
    anet_foo =  get_anet_foo(Omegat.precomputed_arguments, h, mode=1, verbose=False)
    vertex1_array = Omegat.recomputed_arguments['vertex1_array']
    vertex2_array = Omegat.recomputed_arguments['vertex2_array']
    d_array       = Omegat.recomputed_arguments['d_array']
    assert ( anet_foo(vertex1_array, vertex2_array, d_array) is not None )
    assert ( not np.isnan( anet_foo(vertex1_array, vertex2_array, d_array) ).any() ) 

    #test all forces were correctly initialized to zero
    anet = anet_foo(vertex1_array, vertex2_array, d_array)
    assert ( (0 == anet).all() )
    assert ( (0 == anet_vertex1).all() )
    assert ( (0 == anet_vertex2).all() )

    #test that dipslacement, velocity, and acceleration fields are not nan
    assert ( not np.isnan(Omegat.u).any() ) 
    assert ( not np.isnan(Omegat.ud).any() ) 
    assert ( not np.isnan(Omegat.udd).any() ) 

    #test distance_3D
    a = np.array(range(3),dtype=np.float32)
    assert ( np.isclose(distance_3D(a, a+1) , np.sqrt(3) ) )

    # test edges_unique_length
    vertices1 = Omegat.recomputed_arguments['vertex1_array']
    vertices2 = Omegat.recomputed_arguments['vertex2_array']
    d_array   = edges_unique_length(vertices1, vertices2)
    assert ( d_array.shape == (756,) )

    #test that the time_step is not returning any nan values
    time_step = get_time_step(Omegat.precomputed_arguments, h=get_h(h=0.01), mode=1, verbose=False, njitQ=True)
    retval = time_step(vertex1_array, vertex2_array, d_array, u, ud, udd, h)
    for r in retval:
        assert ( not np.isnan(r).any() ) 

    #test stepping forward
    nsteps = 100
    h = get_h(h=0.01)
    u  = Omegat.u
    ud = Omegat.ud
    udd= Omegat.udd
    vertex1_array = Omegat.recomputed_arguments['vertex1_array']
    vertex2_array = Omegat.recomputed_arguments['vertex2_array']
    d_array   = edges_unique_length(vertex1_array, vertex2_array) 

    step_forward_n_times = get_step_forward_n_times(Omegat.precomputed_arguments, h, mode=1, verbose=True, njitQ=True)

    vertices1, vertices2, d_array, u, ud, udd, h = step_forward_n_times(vertex1_array, vertex2_array, d_array, u, ud, udd, h, nsteps)
    tme += h*nsteps

    #test that dipslacement, velocity, and acceleration fields are not nan
    assert ( not np.isnan(u).any() ) 
    assert ( not np.isnan(ud).any() ) 
    assert ( not np.isnan(udd).any() ) 

    Omegat.u   = u
    Omegat.ud  = ud
    Omegat.udd = udd
    Omegat     = update_mesh(Omegat,u)
    assert (Omegat)
    return True




if __name__=='__main__':
	os.chdir(f'../Data/spherical_meshes')
	mesh = trimesh.load('spherical_mesh_64.stl'); 
	#^this line is the one printing.  I don't know how to stop it.
	print('ignore ^that')
	#subtract the center of mass
	mesh.vertices -= mesh.center_mass
	#normalize the mean radius to 1
	mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))

	#compute vertex masses for the displacement invariant barycentric discretization of a 2-D surface.
	mass_array = np.array ( [get_mass(vid, mesh, density = 1.) for vid in range(mesh.vertices.shape[0])] )

	assert (test_initialize_system(mesh, mass_array))
	Omegat = initialize_system(mesh, mass_array)
	assert (test_spring_exists())
	assert (test_spring_add_remove(Omegat))
    assert (test_simplest_controller_at_equilibrium())
	print ('test_controller passed.  Yay!')

