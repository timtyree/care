from numba import njit
import numpy as np, pandas as pd
@njit
def spring_force(x, x0, k):
	'''general spring force in the direction of x with magnitude k*(np.linalg.norm(x) - x0).
	k and x0 are floats. x is a vector.'''
	absx = np.linalg.norm(x)
	dx = absx-x0
	return k*dx*x/absx

#@njit
def get_A(vid, mesh):
	q = np.array(mesh.vertices[vid])
	N1q = np.array(mesh.vertices[mesh.vertex_neighbors[vid]])
	return calc_area(q,N1q)

def angle_between(v1,v2): 
	return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))*180/np.pi

def get_Q_star(vid, mesh, q0=(0,0,0)):
	q  = mesh.vertices[vid]
	N1q = np.array(mesh.vertex_neighbors[vid])
	N1q = N1q[N1q>-1]
	d_star = np.add.reduce([qj-q for qj in mesh.vertices[N1q]])
	Q_star = np.dot(q-q0, d_star)
	return float(Q_star)

def get_Q(mesh, q0=(0,0,0)):
	return np.add.reduce([get_Q_star(vid=vid, mesh=mesh, q0=q0) for vid in range(mesh.vertices.shape[0])])

@njit
def flow_map(q,p,net_force, mass):
	dqdt = p/mass
	dpdt = net_force
	return dqdt, dpdt

#too slow. this takes 2.8 seconds to compute the net forces on 806 rows.  Not ideal
def get_Fs(mesh, df):
	net_spring_forces = 0.*mesh.vertices
	for rowid, row in df.iterrows():
		b, a, k, x0 = row[['b','a','k','x0']]
		b = int(b); a = int(a)
		F = spring_force(x=mesh.vertices[b] - mesh.vertices[a],x0=x0,k=k)
		net_spring_forces[a] -= F
		net_spring_forces[b] += F
	return net_spring_forces

def get_Fp_mean(vertices, VN, pressure_forces, mesh, P):
    '''returns pressure forces in units of gcm/s^2 using the mean A and mean normal'''
    for vid in range(vertices.shape[0]):
        q = vertices[vid]
        N1q = vertices[VN[vid]]
        Fp_magnitude = P*calc_area(q, N1q)
        pressure_forces[vid] = Fp_magnitude * mesh.vertex_normals[vid]

def calc_area_mean(q, N1q):
    '''compute mean area'''
    q_degree = len(N1q)
    q_A = 0. #scalar output * q
    for i in range(q_degree - 1, -1, -1):
        c = N1q[i] -N1q[i-1]
        d = N1q[i-1] - q
        q_A += np.linalg.norm(cross_product(N1q[i] - N1q[i-1], N1q[i-1]))
    if q_degree == 0:
    	return 0
    else:
	    q_A /= (2 * 3 * q_degree)
    return q_A


#using my Astar method ~50ms estimated runtime for 1500 mesh points
def get_Fp(vertices, VN, pressure_forces, P):
    '''returns pressure forces in units of gcm/s^2'''
    for vid in range(vertices.shape[0]):
        q = vertices[vid]
        N1q = vertices[VN[vid]]
        pressure_forces[vid] = P*calc_area(q, N1q)
@njit
def calc_area(q, N1q):
    '''compute A_star'''
    #     d_lst = N1q - q
    q_degree = len(N1q)
    q_A = 0. * q
    for i in range(q_degree - 1, -1, -1):
        c = N1q[i] -N1q[i-1]
        d = N1q[i-1] - q
        q_A += cross_product(N1q[i] - N1q[i-1], N1q[i-1])
    q_A /= (2 * 3)
    return q_A


#this worked with njit
# assert((njit_cross_product(a,b)==np.cross(a,b)).all())
@njit
def cross_product(a, b):
    x = a[1]*b[2]-a[2]*b[1]
    y = a[2]*b[0]-a[0]*b[2]
    z = a[0]*b[1]-a[1]*b[0]
    return np.array((x,y,z))

@njit
def get_spring_coeff(qi,qj, x0, k):
    #compute the stiffness matrix, K
    d  = np.linalg.norm(qj-qi)
    if d==0:
        return 1.
    spring_coeff = k*(d-x0)/d
    return spring_coeff

# @njit
def calc_spring_mat(spring_mat, vertices, x0_constant_mat, k_constant_mat, VN):
    N = len(VN)
    for i in range(N):
        for j in VN[i]:
            qi = vertices[i]
            qj = vertices[j]
            spring_coeff = get_spring_coeff(qi,qj, x0_constant_mat[i,j], k_constant_mat[i,j])
            spring_mat[i][j] = spring_coeff
            spring_mat[i][i] -= spring_coeff
        spring_mat[i][i] = np.add.reduce(spring_mat[i])

def get_mass(vid, mesh, density = 0.105 ):
    '''mu is the density, for cardiac tissue, .105 g/cm^2'''
    # mu = 0.105 #g/cm^2 
    mu = density
    #assume locations are in units of cm
    star = mesh.vertex_faces[vid]
    star = star[star>-1]
    mass = mu/3*np.add.reduce(mesh.area_faces[star])
    return mass #vertex mass in grams

@njit
def mean_magnitudes(forces1,forces2):
    f1tot = 0.; f2tot=0.
    for i in range(forces1.shape[0]):
        f1tot += np.linalg.norm(forces1[i])
        f2tot += np.linalg.norm(forces2[i])
    return f1tot, f2tot

