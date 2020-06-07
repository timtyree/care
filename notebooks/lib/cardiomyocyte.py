#!/bin/bash/env python3
# cardiomyocyte.py
#LOCAL CARDIOMYCYTE CELL RULES FOR TIMESTEPPING ASYNCHRONOUSLY
#- add time_step function to async_queue calling the next item in priority_heap
#- after calling time_step, add that cell to a unit
from numba import njit, jit
import numpy as np

#TODO: find the distribution of triangle area's for the watertight RA mesh I made yesterday.
#@njit
def Tanh(x):
	'''fast/simple approximatation of the hyperbolic tangent function'''
	if ( x < -3.):
		return -1.
	elif ( x > 3. ):
		return 1.
	else:
		return x*(27.+x*x)/(27.+9.*x*x)

# /*------------------------------------------------------------------------
#  * applying periodic boundary conditions for each texture call
#  *------------------------------------------------------------------------
#  */
#@njit
def pbc(x,y, width  = 512, height = 512):
	'''by default, x is in [0,512] and y is in [0,512] at steps of size 1.
	 width = 512 and height = 512
	(x, y) pixel coordinates of texture with values 0 to 1.
	tight boundary rounding is in use.'''
	if ( x < 0  ):				# // Left P.B.C.
		x = width - 1
	elif ( x > (width - 1) ):	# // Right P.B.C.
		x = 0
	if( y < 0 ):				# //  Bottom P.B.C.
		y = height - 1
	elif ( y > (height - 1)):	# // Top P.B.C.
		y = 0
	return x,y

#@njit
def _get_distance_2D_pbc(point_1, point_2, width  = 512, height = 512):
	'''assumes getting shortest distance between two points with periodic boundary conditions in 2D '''
	return _get_distance_ND_cartesian_pbc(point_1, point_2, (width, height))
#@njit
def _get_distance_ND_cartesian_pbc(point_1, point_2, mesh_shape):
    '''assumes getting shortest distance between two points with periodic boundary conditions '''
    dq2 = 0.
    for q1, q2, width in zip(point_1, point_2, mesh_shape):
        dq2 += np.min(((q2 - q1)**2, (q2 + width - q1 )**2))
    return np.sqrt(dq2)

#@njit
def _get_distance_ND_cartesian(point_1, point_2, mesh_shape):
    '''assumes getting shortest distance between two points with periodic boundary conditions '''
    dq2 = 0.
    for q1, q2, width in zip(point_1, point_2, mesh_shape):
        dq2 += (q2 - q1)**2
    return np.sqrt(dq2)

#@njit
def get_distance(point_1, point_2):
	'''return the 2D distance between two points using periodic boundary conditions'''
	return _get_distance_2D_pbc(point1, point_2)

# TODO: define the cardiomyocyte class
#@njit
class CardioMyocyte(object)
    def __init__(self, **kwargs):
    	'''a triangular patch of myocardial tissue.

    	    a more extensive description:
		   
		    Key Word Arguments
		    ----------
		    data       : for example, a numpy array, pandas df, a spark rdd, or a rapids rmm.
		    	stores data 
		    triangle   : list of floats, for example, [0.123, 0.456, 10.4321]
		        cartesian coordinates of vertices
		    face_id    : int, for example, 576
		    vertex_id  : int, for example, 1234
		    group_id   : int, -1 indicates type/phase of cell 

		    # Nota Bene: a continuous phase_field may be implemented and is worth trying!
		    
		'''
        #configuration properties
        self.data        = kwargs['data' ]    
        self.trngle      = kwargs['triangle' ] 
        self.face_id     = kwargs['face_id'  ] 
        self.vertex_id   = kwargs['vertex_id'] 
        self.group_id    = kwargs['group_id' ] 
        self.area = self.calc_area()
        #default local state properties.  Repolarized tissue is excitable.
        self.voltage   = 0.0
        self.fast      = 1.0
        self.slow      = 0.4

    #######################################
    # Functionality for a local 2D Geometry
    #######################################
    #@njit
    def calc_area(self):
    	normal = self.get_normal()
    	return np.linalg.norm(normal)

    #@njit
    def get_normal(self):
    	e1 = self.triangle[0]
    	e2 = self.triangle[1]
    	return np.cross(e1,-e2)

	#@njit
	def get_unit_normal(self):
		return self.get_normal()/np.linalg.norm(self.get_normal())

	#@njit 
	def get_degree_matrix_element(self):
		return len(self.get_neighboring_ids())

	#@njit
	def get_adjacency_matrix_element(self, neighboring_ids):
		return 

    ########################################
    #  Various Graph Laplacians
    ########################################
	# for descriptions, see https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_for_simple_graphs
	#@njit
	def get_laplacian(self, neighboring_ids):
		return _get_graph_laplacian(self, neighboring_ids)
		    
    #@njit
    def _get_graph_laplacian(self, neighboring_ids):
    	'''return the graph laplacian of self with neighbors neighboring_ids'''
    	d2Vdx2 = self.get_degree()
    	for nid in neighboring_ids:
    		d2Vdx
    	return (d2Vdx2, 0., 0.)

    ########################################
    #  Functionality for Integrating in Time
    ########################################
    #@njit
    def pass_params_to_models(**kwargs):
    	'''return foo, goo, where foo is the local state time step and goo is the local shape time step.'''
    	foo = time_step_at_state_at_pixel(**kwargs)
		goo = time_step_at_shape_at_pixel(**kwargs)
    	return foo, goo

    #@njit
    def get_local_term(self):
    	'''TODO: dsdt[0] = -I_ion/C_m'''
    	dsdt = (dVdt, dfastdt, dslowdt)
    	return dsdt

    #@njit
    def get_nonlocal_term(self):
    	'''TODO: dsdt[0] = D * laplacian of self'''
    	dsdt = (dVdt, dfastdt, dslowdt)
    	return dsdt

    #@njit
    def time_step_state_at_pixel(self, h, neighboring_ids):
    	local_term    = self.get_local_term()
    	nonlocal_term = self.get_nonlocal_term(neighboring_ids)
    	# use forward euler integration through time
    	self.state += h * ( local_term + nonlocal_term )
    	return self

    #@njit
    def time_step_at_shape_at_pixel(self, h, neighboring_ids):
    	#TODO: define mechanical parameters from kwargs
    	neighboring_ids   = self.get_neighboring_ids()
    	
    	#TODO: make this call asynchronous when all neighbors are at least up to date with vertex_id
    	if self.neighbors_up_to_date(vertex_id, neighboring_ids):


    	#TODO: after asynchronous call is completed, update time
    	#TODO
    	return self

    #@njit
def time_step_at_pixel(inVfs, x, y):#, h):
	# define parameters
	width  = 512
	height = 512
	ds_x   = 5 #18 #domain size
	ds_y   = 5 #18

	# dt = 0.1
	diffCoef = 0.001
	C_m = 1.0

	#no spiral defect chaos observed for these parameters (because of two stable spiral tips)
	# tau_pv = 3.33
	# tau_v1 = 19.6
	# tau_v2 = 1000
	# tau_pw = 667
	# tau_mw = 11
	# tau_d  = 0.42
	# tau_0  = 8.3
	# tau_r  = 50
	# tau_si = 45
	# K      = 10
	# V_sic  = 0.85
	# V_c    = 0.13
	# V_v    = 0.055
	# C_si   = 1.0
	# Uth    = 0.9


	def set_params(self, **kwargs):
		#TODO: update global variables

		#update dynamical variables
		params = ('tau_pv', 'tau_v1', 'tau_v2', 
			'tau_pw', 'tau_mw', 'tau_d', 'tau_0', 'tau_r', 'tau_si',
			)
		if (param in kwargs): self.a = kwargs['a']

	    self.params.tau_pv   = kwargs['' ]    
	    self.params.tau_v1   = kwargs['' ] 
	    self.params.tau_v2   = kwargs['' ] 
	    self.params.tau_pw   = kwargs['' ] 
	    self.params.tau_mw   = kwargs['' ]
	    self.params.tau_mw   = kwargs['' ]
	    self.params.tau_mw   = kwargs['' ]     
		#these parameters supported spiral defect chaos beautifully
	struct_dtype = np.dtype([(param, np.float64) for param in params])

	tau_pv = 3.33
	tau_v1 = 15.6
	tau_v2 = 5
	tau_pw = 350
	tau_mw = 80
	tau_d = 0.407
	tau_0 = 9
	tau_r = 34
	tau_si = 26.5
	K = 15
	V_sic = 0.45
	V_c = 0.15
	V_v = 0.04
	C_si = 1
	Uth = 0.9
    ######################################
    # Getter & Setter for Cell State/Shape 
    # Caution! These might break immutability needed for some LLVM compilation.  Not sure.
    ######################################
    @property
    def triangle(self):
        return self.trngle
    
    @triangle.setter
    def triangle(self, new_triangle):
    	if not len(new_triangle) == 3:
            raise ValueError("Error: triangle must be have three vertices, not {0}!".format(len(new_triangle)))
        self.trngle = new_triangle

    @property
    def state(self):
    	'''return (self.voltage, self.fast, self.slow)'''
        return (self.voltage, self.fast, self.slow)
    
    @state.setter
    def state(self, new_state):
    	if not len(state) == 3:
            raise ValueError(f"Error: expected number of state channels is {3}, not {len(new_state)}!")
        self.state = new_state

