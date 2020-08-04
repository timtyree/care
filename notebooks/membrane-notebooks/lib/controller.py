#initialization for controller.py
import numpy as np, pandas as pd

#automate the boring stuff
from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()
# sys.path.append("../lib") 
# from  import *
# sys.path.append("lib") 
# from lib import *
from vertex_shader import *
from spring import *

# from operari import *
# from ProgressBar import *
# from mesh_ops import *

# the visualization tools involved here for triangular meshes is
import trimesh
# import pyglet
from numba import njit, cuda
# from numba.typed import List
# import numba


####################################################################################################
# TODO List
####################################################################################################
# TODO: start a test_controller.py function and record passing tests in it for each function herein

# TODO: debug get_time_step until python can interpret it
# retval(ud, vertex1_array, vertex2_array, d_array)
#         anet_array_foo = get_anet_array_foo (vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)

# foo = retval(ud, vertex1_array, vertex2_array, d_array)
# foo2= foo(ud, vertex1_array, vertex2_array, d_array)
# (ud, vertex1_array, vertex2_array, d_array)



#compute the first time step
# TODO: update Omegat, time
# TODO: test ^this with a simple use case
# TODO: njit ^this!!!

#TODO: did I already njit anet_foo?
#TODO(later): does JAX give an easy way to differentiate wrt Omegat.vertices?

# TODO: repeated forward time step n times
# TODO: test ^this with a simple use case
# TODO: njit ^this!!!

# TODO: compute Vol, SVR timeseries
# TODO: get_dudt 
# TODO: get_fnet
# TODO: compute_masses of nodes
# TODO: initialize_
# TODO: briefly try find get_time_step outline?
# TODO: implement forward euler integration time_step()
# TODO(later): method to go to/from recomputed_arguments and u,ud,udd


####################################################################################################
# Functional Goal: time step the mechanics n times 
####################################################################################################
def time_step_n_times(x,X,h,n):
	for k in range(n):
		x += time_step(x,X,h)
	return x
def get_h(h = 10**-2):
	'''constant time steps'''
	return h
# def get_h(h = 10**-2, beta = 1., acceptedQ=True):
#     '''exponential adaptive time steps
#     h = most recent time step,
#     acceptedQ = whether the most recent time step was accepted,
#     beta = stepsize change in parameter.
#     **caution** this get_h could be unstable/inefficient with any unbiased R.W..
#     Use it with steps directed towards the minimizer
#     '''
#     return h

# #Example Usage: 
# mesh.vertices = time_step_n_times(
#     mesh.vertices,
#     X,
#     get_h(),
#     n=1)

####################################################################################################
# Handling configuration/mesh class before/after [ compiled ] time integration
####################################################################################################
def precompute_mesh(mesh, mass_array):
	'''
	precompute edges, which are the force generating simplices here

	 mass_array is an array of masses to assign at each node, for example
	 #compute vertex masses for the displacement invariant barycentric discretization of a 2-D surface.
	mass_array = np.array ( [get_mass(vid, mesh, density = 1.) for vid in range(mesh.vertices.shape[0])] )
	'''
	
	Omega0 = mesh.copy()
	Omegat = Omega0.copy()
	meu = mesh.edges_unique

	vid1_array, vid2_array = np.array(meu[:,0],dtype=int), np.array( meu[:,1],dtype=int)
	vertex1_array = np.array ( mesh.vertices[vid1_array]  , dtype=float) 
	vertex2_array = np.array ( mesh.vertices[vid2_array]  , dtype=float) 
	vertex_array =  np.array(Omega0.vertices, dtype=float) #only needed for shape
	
	
	#initialize spring force parameters
	d0_array = np.array(mesh.edges_unique_length, dtype=float)
	k0_array = np.ones_like(vertex1_array[...,0], dtype=float)
	#f0_array = np.array(np.zeros_like(mesh.vertices, dtype=float))

	#collect precomputed_arguments into a dict.
	precomputed_arguments = {
		'd0_array':d0_array, 
		'k0_array':k0_array, 
		'vid1_array':vid1_array, 
		'vid2_array':vid2_array, 
		'vertex_array':vertex_array,
		'mass_array':mass_array
	 }

	#compute the recomputed arguments into a dict.
	# TODO: move this into its own function and call both from initialize_system
	vertex1_array = np.array( mesh.vertices[vid1_array],dtype=float)
	vertex2_array = np.array( mesh.vertices[vid2_array],dtype=float)
	d_array       = np.array(mesh.edges_unique_length  , dtype=float)
	recomputed_arguments = {
		 'vertex1_array':vertex1_array, 
		'vertex2_array':vertex2_array, 
		'd_array':d_array, 
	 }
	Omegat.precomputed_arguments = precomputed_arguments.copy()
	Omegat.recomputed_arguments  =  recomputed_arguments.copy()
	Omegat.X = vertex_array.copy()
	Omegat.Omega0 = Omega0.copy()
	return Omegat
def update_mesh(Omegat,u):
	return compute_mesh_update(Omegat,u)

def compute_mesh_update(Omegat,u):
	'''Omegat is the configuration, u is the displacement since t=0
	note that Omegat.copy() should not be returned, 
	since ^this will wipe the precompute_mesh output, which somewhat defeats its purpose
	'''
	#TODO(later): check whether Omegat.Omega0 exists and if it doesn't run precompute_mesh(Omegat,mass_array)
	#compute the spring deformation states for each edge
	
	vid1_array = Omegat.precomputed_arguments['vid1_array']
	vid2_array = Omegat.precomputed_arguments['vid2_array']
	vertex_array = Omegat.precomputed_arguments['vertex_array']
	#update material vertices with material displacement field u
	Omegat.vertices = Omegat.Omega0.vertices + u 
	# nota bene: ^this should not be Omegat.vertices = Omegat.X + u 
	#     (Omegat.X removes class TrackedArray from trimesh, which is made use of in Omegat.edges_unique_length)
	
	#compute the spring deformation states for each edge
	vertex1_array = np.array( Omegat.vertices[vid1_array],dtype=float)
	vertex2_array = np.array( Omegat.vertices[vid2_array],dtype=float)
	d_array = np.array(Omegat.edges_unique_length, dtype=float)
	
	recomputed_arguments = {
		'vertex1_array':vertex1_array, 
		'vertex2_array':vertex2_array, 
		'd_array':d_array, 
	 }
	
	
	Omegat.recomputed_arguments  =  recomputed_arguments.copy()	
	return Omegat

def initialize_system(mesh, mass_array):
	''' precompute fields related to X = the material-space configuration
	and	initialize system fields
	u   = the displacement field
	ud  = the velocity field
	udd = the acceleration field.
	'''
	Omegat = precompute_mesh(mesh, mass_array)
	u  = np.zeros_like(Omegat.X, dtype=float)
	Omegat = compute_mesh_update(Omegat,u)
	Omegat = precompute_mesh(mesh, mass_array)
	#start the system from rest
	ud = u.copy()
	udd= u.copy()
	Omegat.u  = u
	Omegat.ud = ud
	Omegat.udd= udd
	return Omegat



####################################################################################################
# Get/compile/precompute the time step function
####################################################################################################
def get_time_step(precomputed_arguments, h, mode=1, verbose=True):
	return get_anet_foo(precomputed_arguments=precomputed_arguments, h=h, mode=mode, verbose=verbose)

def get_anet_foo(precomputed_arguments, h, mode=1, verbose=True):
	'''TODO: define the simplest productive time step 
	- unbiased R.W. that only accepts local improvements.
	- 1. FEI directed by elastic model.
	- 2. FEI directed by spring model.
	- 3. Newmark directed by elastic model.
	- 4. Newmark directed by spring model.
	- R.W. that only accepts local improvements, with steps directed spring/elastic model.
	TODO: define an array of njit'd time_step(h) functions for h in step_size_array
	- '''
	#         def get_forward_euler_integration_time_step(u, ud, X, anet_array, h, 
	#                                                     d0_array, k0_array, vid1_array, vid2_array, vertex_array):
	if mode==1:
		#precomputed model parameters don't need to be passed everytime to the time_step abstraction
		d0_array = precomputed_arguments['d0_array']
		k0_array = precomputed_arguments['k0_array']
		vid1_array  = precomputed_arguments['vid1_array']
		vid2_array  = precomputed_arguments['vid2_array']
		vertex_array  = precomputed_arguments['vertex_array']
		mass  = precomputed_arguments['mass_array']
		#arguments of the function returned
		retargs = ['h', 'vertex1_array', 'vertex2_array', 'd_array']
		if verbose:
			print ('explicit forward euler integration is used in the time_step function returned.')
			print (f"Example usage: u,ud = time_step(")
			for s in retargs:
				print (f"\t{s},")
			print (")")

		# def get_anet_array_foo (d0_array, k0_array, vid1_array, vid2_array, vertex_array):
			'''compute the net spring force for each vertex.'''
		def anet_foo(vertex1_array, vertex2_array, d_array):
			'''compute the net spring force for each vertex.'''
			#TODO(later): speed up anet_foo by making anet_vertex1, anet_vertex2 be returned directly from compute_spring_forces
			anet = compute_spring_forces ( 
				vertex1_array, vertex2_array, d_array, 
				d0_array, k0_array, vid1_array, vid2_array, vertex_array
				)
			# anet_vertex1 = anet[vid1_array]
			# anet_vertex2 = anet[vid2_array]
			return anet
			# return np.divide( anet , mass_array ) 
		return njit(anet_foo)
	else:
		raise(f"Error! Method note implemented!")
		print('this mode is not yet implemented in controller.get_anet_foo')
		return False

# f0_array_foo   = lambda vertex1_array, vertex2_array, d_array: compute_spring_forces ( vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)
#             fnet_array_foo = f0_array_foo  # + any other forces desired
#             anet_array_foo = np.divide ( fnet_array_foo , mass_array ) 
#             anet_array_foo = lambda vertex1_array, vertex2_array, d_array: np.divide ( fnet_array_foo ( vertex1_array, vertex2_array, d_array) , mass_array ) 
# return f0_array_foo 

# 		def get_anet_array_foo (vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array):
# 			'''compute the net spring force for each vertex.'''
# 			f0_array_foo   = lambda vertex1_array, vertex2_array, d_array: compute_spring_forces ( vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)
# #             fnet_array_foo = f0_array_foo  # + any other forces desired
# #             anet_array_foo = np.divide ( fnet_array_foo , mass_array ) 
# 			#             fnet_array_foo = lambda vertex1_array, vertex2_array, d_array: f0_array_foo ( vertex1_array, vertex2_array, d_array) # + any other forces desired
# 			#             anet_array_foo = lambda vertex1_array, vertex2_array, d_array: np.divide ( fnet_array_foo ( vertex1_array, vertex2_array, d_array) , mass_array ) 
# 			return f0_array_foo 
# anet_array_foo = get_anet_array_foo (d0_array, k0_array, vid1_array, vid2_array, vertex_array)
# anet_array_foo = lambda vertex1_array, vertex2_array, d_array: np.divide ( anet_array_foo(vertex1_array, vertex2_array, d_array) , mass_array )
# return anet_array_foo
# #         @njit
# 		def get_d2udt2 ( anet_array_foo, vertex1_array, vertex2_array, d_array ):
# 			return anet_array_foo
# #         @njit
# 		def get_dudt (h, anet_array_foo):
# 			'''returnds dudt.'''		
# 			return lambda ud, vertex1_array, vertex2_array, d_array: ud + h * get_d2udt2 ( anet_array_foo, vertex1_array, vertex2_array, d_array )
# #             dudt  = lambda ud, vertex1_array, vertex2_array, d_array: ud + h * udd ( vertex1_array, vertex2_array, d_array )
# #             dudt  = ud + h * udd ( vertex1_array, vertex2_array, d_array )
# #             return dudt
# #         @njit
# 		def get_stepper (h, u, ud, anet_array_foo):
# 			'''example usage: 
# 			time_step = get_time_step (h, u, ud, anet_array_foo) 
# 			u_new = time_step(u, ud, vertex1_array, vertex2_array, d_array)
# 			TODO: write functional controller methods get_forward_integrate_n_steps
# 			TODO: minimalist test cases.  debug until it works
# 			TODO: njit this
# 			'''
# 			#                 dudt  = lambda ud, vertex1_array, vertex2_array, d_array: get_dudt (h, ud, anet_array_foo )
# 			dudt  = get_dudt (h, anet_array_foo )
# #             time_step = lambda u, ud, vertex1_array, vertex2_array, d_array : u + np.multiply(h , get_dudt (h, ud, anet_array_foo ))
# #             time_step = lambda u, ud, vertex1_array, vertex2_array, d_array : u + h * dudt (ud, vertex1_array, vertex2_array, d_array)
# #             def time_step(u, ud, vertex1_array, vertex2_array, d_array ) : 
# #                 return u + h * dudt (ud, vertex1_array, vertex2_array, d_array)
# #             return dudt
# 			return dudt
# #             return time_step, dudt
# 		return get_stepper (h, u, ud, anet_array_foo )   
# #         time_step, dudt =  get_stepper (h, u, ud, anet_array_foo ) 
# #         return time_step(u, ud, vertex1_array, vertex2_array, d_array)
# 	else:
# 		raise(f"Error! Method note implemented!")


#TODO: debug get_anet_array_foo until python can interpret it
def get_anet_array_foo (vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array):
	'''compute the net spring force for each vertex.'''
	f0_array_foo   = lambda vertex1_array, vertex2_array, d_array: compute_spring_forces ( vertex1_array, vertex2_array, d_array, d0_array, k0_array, vid1_array, vid2_array, vertex_array)
#             fnet_array_foo = f0_array_foo  # + any other forces desired
#             anet_array_foo = np.divide ( fnet_array_foo , mass_array ) 
	#             fnet_array_foo = lambda vertex1_array, vertex2_array, d_array: f0_array_foo ( vertex1_array, vertex2_array, d_array) # + any other forces desired
	#             anet_array_foo = lambda vertex1_array, vertex2_array, d_array: np.divide ( fnet_array_foo ( vertex1_array, vertex2_array, d_array) , mass_array ) 
	return f0_array_foo 

####################################################################################################
# Manipulation of springs/force generating edges with two vertices
####################################################################################################
def spring_exists(vid1_array, vid2_array, vid1,vid2):
	'''check where an edge exists'''
	return sum ( (vid1_array == vid1) & (vid2_array == vid2) ) > 0

def remove_springs(vid1_array, vid2_array, blacklist):
	'''
	remove a list of springs (blacklist) from the list of springs, (vid1_array,vid2_array) 
	found in Omegat.precomputed_arguments.
	
	vid1_array is an array of ints
	vid2_array is an array of ints
	blacklist is any input list of edges as a list of tuples of ints
	'''
	v1_list = []
	v2_list = []
	drop = (0.*vid1_array).astype(bool)
	for ax, bx in blacklist:
		boo   = (vid1_array == ax) & (vid2_array == bx) 
		drop |= boo
	return vid1_array[~drop],vid2_array[~drop]

def add_springs(vid1_array, vid2_array, whitelist):
	'''
	add a list of springs (blacklist) from the list of springs, (vid1_array,vid2_array) 
	found in Omegat.precomputed_arguments.
	
	vid1_array is an array of ints
	vid2_array is an array of ints
	blacklist is any input list of edges as a list of tuples of ints
	'''
	v1_list = list(vid1_array)
	v2_list = list(vid2_array)
	for ax, bx in whitelist:
		if not spring_exists(vid1_array, vid2_array, vid1=ax,vid2=bx):
			v1_list.append(ax)
			v2_list.append(bx)
	return np.array(v1_list, dtype=int), np.array(v2_list, dtype=int)

@njit
def distance_3D(a, b):
    '''a euclidian distance function in 3D real space'''
    out  = (a[0]-b[0])**2
    out += (a[1]-b[1])**2
    out += (a[2]-b[2])**2
    return np.sqrt(out)

@njit
def edges_unique_length(vertices1, vertices2):
    '''returns the distances between vertices1 and vertices2'''
    N = vertices1.shape[0]
    d_array = np.zeros(N,dtype=np.float32)
    for n in range(N):
        d_array[n] = distance_3D(vertices1[n], vertices2[n])
    return d_array 

def get_time_step(precomputed_arguments, h, mode=1, verbose=True, njitQ=True):
    anet_foo = get_anet_foo(precomputed_arguments=precomputed_arguments, h=h, mode=mode, verbose=verbose)
    vid1_array = precomputed_arguments['vid1_array']
    vid2_array = precomputed_arguments['vid2_array']
    X          = precomputed_arguments['vertex_array'] #X = material coordinates

    def time_step(vertex1_array, vertex2_array, d_array, u, ud, udd, h):
        #TODO(speed up time_step): move from edge basis to vertex basis only after all time steps occurred 
        anet = anet_foo(vertex1_array, vertex2_array, d_array)
        udd = anet
        ud = ud + h*udd
        u  = u + h*ud
        x  = X + u
        
        #update vertex1_array, vertex2_array, d_array
        vertices1 = x[vid1_array]
        vertices2 = x[vid2_array]
        d_array = edges_unique_length(vertices1, vertices2)
        return vertices1, vertices2, d_array, u, ud, udd, h
    
    if njitQ:
        return njit(time_step)
    else:
        return time_step

def get_step_forward_n_times(precomputed_arguments, h, mode=1, verbose=True, njitQ=True):
    '''returns forward euler integration, get_step_backward_n_times(vertex1_array, vertex2_array, d_array, u, ud, udd, h, nsteps), which 
    returns its updated arguments, vertex1_array, vertex2_array, d_array, u, ud, udd, h'''
    time_step = get_time_step(precomputed_arguments=precomputed_arguments, h=h, mode=mode, verbose=verbose, njitQ=njitQ)
    
    def step_forward_n_times(vertex1_array, vertex2_array, d_array, u, ud, udd, h, nsteps):
        for n in range(int(nsteps)):
            vertex1_array, vertex2_array, d_array, u, ud, udd, h = time_step(vertex1_array, vertex2_array, d_array, u, ud, udd, h)
        return vertex1_array, vertex2_array, d_array, u, ud, udd, h
    
    if njitQ:
        return njit(step_forward_n_times)
    else:
        return step_forward_n_times

def get_step_backward_n_times(precomputed_arguments, h, mode=1, verbose=True, njitQ=True):
    '''returns naive backward euler integration, get_step_backward_n_times(vertex1_array, vertex2_array, d_array, u, ud, udd, h, nsteps), which 
    returns its updated arguments, vertex1_array, vertex2_array, d_array, u, ud, udd, h'''
    time_step = get_time_step(precomputed_arguments=precomputed_arguments, h=h, mode=mode, verbose=verbose, njitQ=njitQ)
    
    def step_backward_n_times(vertex1_array, vertex2_array, d_array, u, ud, udd, h, nsteps):
        for n in range(int(nsteps)):
            vertex1_array, vertex2_array, d_array, u, ud, udd, h = time_step(vertex1_array, vertex2_array, d_array, u, ud, udd, -h)
        return vertex1_array, vertex2_array, d_array, u, ud, udd, h
    
    if njitQ:
        return njit(step_backward_n_times)
    else:
        return step_backward_n_times


if __name__=='__main__':
	import os, trimesh, numpy as np
	from spring import *
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

	#specify spring force parameters for contracting all edges by 50%
	vol_frac = 0.5
	d0_array = vol_frac*np.array(mesh.edges_unique_length, dtype=float)
	k0_array = np.ones_like(vertex1_array[...,0], dtype=float)

	Omegat.precomputed_arguments['d0_array'] = d0_array.copy()
	Omegat.precomputed_arguments['k0_array'] = k0_array.copy()

	print(Omegat.volume)

	nsteps = 10000
	h = get_h(h=0.001)
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

	print(Omegat.volume)
	print("did the volume of the sphere decrease? \nThe equilibrium edge length should be half of the initial edge length.")