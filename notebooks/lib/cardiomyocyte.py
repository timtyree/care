#!/bin/bash/env python3
# cardiomyocyte.py
#LOCAL CARDIOMYCYTE CELL RULES FOR TIMESTEPPING ASYNCHRONOUSLY
#- add time_step function to async_queue calling the next item in priority_heap
#- after calling time_step, add that cell to a unit
from numba import njit, jit
from numba.typed import List
import numpy as np

# /*------------------------------------------------------------------------
#  * Mechanica
#  *------------------------------------------------------------------------
#  */

def SignedVolumeOfTriangle(p1, p2, p3):
	'''Vector p1, Vector p2, Vector p3'''
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123)

def SignedVolumeOfTrapezoid(q, qa, qb, q0):
	'''each argument is an iterable of exactly three floats, giving xyz coordinates.'''
	q0x,q0y,q0z = q0
	qx,  qy, qz = q
	ax,  ay, az = qa
	bx,  by, bz = qb
	return ((q0z - qaz) * (-ay * bx + ax * by + ay * qx - by * qx - ax * qy + bx * qy) + (q0y - 
		qay) * (az * bx - ax * bz - az * qx + bz * qx + ax * qz - bx * qz) + (q0x - 
		qax) * (-az * by + ay * bz + az* qy - bz * qy - ay * qz + by * qz))/4.

def VolForFace(q, qa, qb, q0):
	return SignedVolumeOfTrapezoid(q, qa, qb, q0)


def VolumeOfMesh(Mesh mesh):
	'''Mesh mesh'''
	lst = List()
    for vols in mesh.Triangles:
    	lst.append(stSignedVolumeOfTriangle(t.P1, t.P2, t.P3))
    return np.abs(np.sum(lst))



#TODO: find the distribution of triangle area's for the watertight RA mesh I made yesterday.
@njit
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
@njit
def pbc(x,y, width  = 600, height = 600):
	'''by default, x is in [0,600] and y is in [0,600] at steps of size 1.
	 width = 600 and height = 600
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
def _get_distance_2D_pbc(point_1, point_2, width  = 600, height = 600):
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
class CardioMyocyte(object):
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
			vertex_id  : int, for example, [1234, 100, 22]
			group_id   : int, -1 indicates type/phase of cell 

			# Nota Bene: a continuous phase_field may be implemented and is worth trying!
			
		'''
		#configuration properties
		self.data        = kwargs['data'     ]    
		self.trngle      = kwargs['triangle' ] 
		self.face_id     = kwargs['face_id'  ] 
		self.vertices    = kwargs['vertices' ] 
		self.group_id    = kwargs['group_id' ] 
		self.area = self.calc_area()
		#default local state properties.  Repolarized tissue is excitable.
		self.voltage   = 0.0
		self.fast      = 1.0
		self.slow      = 0.4

	def __doc__():
		return '''a triangular patch of myocardial tissue.

			a more extensive description:
		   
			Key Word Arguments
			----------
			data       : for example, a numpy array, pandas df, a spark rdd, or a rapids rmm.
				stores data 
			triangle   : list of floats, for example, [0.123, 0.456, 10.4321]
				cartesian coordinates of vertices
			face_id    : int, for example, 576
			vertex_id  : int, for example, [1234, 100, 22]
			group_id   : int, -1 indicates type/phase of cell 

			# Nota Bene: a continuous phase_field may be implemented and is worth trying!
			'''

	def __repr__(self):
		return f"""Hello! My name is {self.face_id}.")
		\tHave you met my three vertices? They're, {self.vertices}.")
		\tOh, come on in! Meet the family!")
			\there's my area, {self.area:.1f} length^2,
			\there's my voltage, {self.voltage:.3f} units,
			\there's my fast variable, {self.fast:.3f} units, and 
			\there's my slow variable, {self.slow:.3f} units. Come again!
			"""

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
	def get_adjacency_matrix_element(self, other):
		return 

	def get_displacement_matrix_element(point_1, point_2):
		if self.get_adjacency_matrix_element()==1.:
			return _get_distance_2D_pbc(point_1, point_2, width  = 600, height = 600)

	########################################
	#  Various Graph Laplacians
	########################################
	# for descriptions, see https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_for_simple_graphs
	#@njit
	def get_laplacian(self, neighboring_ids):
		return _get_graph_laplacian(self, neighboring_ids)
			
	#@njit
	def _get_neighbor_differences(self, neighboring_ids):
		'''return the graph laplacian of self with neighbors neighboring_id.
		operates on first channel since it`s the only one that's nonlocal.'''
		diff_V_list   = []
		for nid in neighboring_ids:
			other  = self.data.loc[nid]
			point_1, point_2 = (np.mean(self.triangle, axis=0),np.mean(other.triangle, axis=0))
			disp   = get_displacement_matrix_element(point_1, point_2)
			diff_V_list.append( disp )
		return diff_V_list

	def _my_get_laplacian():
		'''Ignore this for now: compute laplaction from local infinitely smooth fit.
		disp_list'''
		disp_lst, angle_list = _get_local_geometry(self, self.get_neighboring_ids())
		total_angle = np.max(angle_lst) - np.min(angle_lst)
		#TODO(see magnus expansion?):  establish the best local 2D coordinates.  suppose locally Lie smoothness.

		#TODO: define an isotropic laplacian with respect to those best local 2D coordinates
		return dV2dx2

	def _get_local_geometry(self, neighboring_ids):
		'''assign angular coordinates to neighbors'''
		theta_lst = [0.]
		self.get_neighboring_ids()
		point_1, point_2 = (np.mean(self.triangle, axis=0),np.mean(other.triangle, axis=0))
		point_old = point_2
		disp   = [self.get_displacement_matrix_element(point_1, point_2)]
		if len(neighboring_ids)==0:
			return disp_lst, angle_list
		for nid in neighboring_ids[1:]:
			other  = self.data.loc[nid]
			point_1, point_2 = (np.mean(self.triangle, axis=0),np.mean(other.triangle, axis=0))
			disp   = self.get_displacement_matrix_element(point_1, point_2)
			angle  = _get_angle(q1=point_1, q2=point_old, q3=point_2)
			theta_lst.append(angle)
		return disp_lst, angle_list

	def _get_angle(q1, q2, q3):
		'''returns the angle between q2-q1 and q3-q1'''
		return np.angle(q2-q1, q3-q1)

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

	def _get_length_of_shared_edge(self,other):
		'''I sped this part up slightly by doing a set comparison on integer triangle index instead of real numbers'''
		vertices = self.vertices
		id1, id2 = tuple(set(vertices).intersection(set(other.vertices)))
		point_lst = []
		for n, q, v in enumerate(zip(self.triangle, vertices):
			if v == id1:
				point_lst.append(q)
			if v == id2:
				point_lst.append(q)
		point_1, point_2 = point_lst
		return float(get_distance(point_1, point_2))
	
	#@njit
	def get_nonlocal_term(self):
		'''returns the transmembrane voltage diffusion term computed by the finite volume method.
		Assumes diffusion tensor is a constant scalar everywhere
		TODO: dsdt[0] = D * laplacian of self'''
		neighboring_ids = self.get_neighboring_ids()
		D = self.diffCoef
		dVdt = 0.
		for nid in neighboring_ids:
			other  = self.data.loc[nid]
			L = _get_length_of_shared_edge(self,other)#length of shared edge
			dVdq = (other.voltage - self.voltage)/get_distance(point_1=np.mean(self.triangle), point_2=np.mean(other.triangle))
			dVdt += L*dVdq
		dVdt *= self.diffCoef
		dfastdt = 0.; dslowdt = 0.;
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
		if self.neighbors_up_to_date(neighboring_ids):
			pass

		#TODO: after asynchronous call is completed, update time
		#TODO
		return self

	#@njit
def time_step_at_pixel(inVfs, x, y):#, h):
	# define parameters
	width  = 600
	height = 600
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
		'''numba wants structs like this:
			struct_dtype = np.dtype([(param, np.float64) for param in params])'''
		#update global variables
		params = ('width', 'height', 'ds_x', 'ds_y', 'diffCoef', 'C_m')
		for param in params:
			if (param in kwargs): self.params[param] = kwargs[param]

		#update dynamical variables
		params = ('tau_pv', 'tau_v1', 'tau_v2', 
			'tau_pw', 'tau_mw', 'tau_d', 'tau_0', 'tau_r', 'tau_si',
			'K', 'V_sic', 'V_c', 'V_v', 'C_si', 'Uth')
		for param in params:
			if (param in kwargs): self.params[param] = kwargs[param]


	struct_dtype = np.dtype([(param, np.float64) for param in params])

	# tau_pv = 3.33
	# tau_v1 = 15.6
	# tau_v2 = 5
	# tau_pw = 350
	# tau_mw = 80
	# tau_d = 0.407
	# tau_0 = 9
	# tau_r = 34
	# tau_si = 26.5
	# K = 15
	# V_sic = 0.45
	# V_c = 0.15
	# V_v = 0.04
	# C_si = 1
	# Uth = 0.9
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

