import numpy as np
##########################################################################################
# Geometric Functions for rotating/deforming/aligning triangles.  
# For the purpose of measuring deformation gradient between two triangles 
# Tim Tyree 
# 6.25.2020
##########################################################################################

##########################################################################################
# #Example Usage: test the explicit deformation map for a one triangle to another triangle
# import trimesh
# mesh = trimesh.load('../Data/spherical_meshes/spherical_mesh_64.stl')
# #subtract the center of mass
# mesh.vertices -= mesh.center_mass
# #normalize the mean radius to 1
# mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))
# # test the explicit deformation map for a number of triangles
# tris = mesh.triangles[71]
# trim = mesh.triangles[30]
# mtos = get_phi(trim,tris)
# trim_mapped = np.array([mtos(trim[0]),mtos(trim[1]),mtos(trim[2])])
# print('tris is')
# print(tris)
# print('trim is mapped to')
# print(trim_mapped)
# print('difference after mapping is')
# print(tris - trim_mapped)
# assert(np.isclose(tris - trim_mapped,0.).all())
##########################################################################################



##########################################################################################
# Rotations
##########################################################################################

#TODO: optimize this function by removing the first two lines (and maybe precomputing a?), and then njiting it.
def rotation_matrix(axis, theta):
	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.  Uses the Euler-Rodriguez formula.
	"""
	axis = np.asarray(axis)
	axis = axis / np.sqrt(np.dot(axis, axis))
	a = np.cos(theta / 2.0)
	b, c, d = -axis * np.sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_shape(triangle):
	d1 = triangle[1]-triangle[0]     #TODO: precompute in final algorithm for material space triangles
	d2 = triangle[2]-triangle[0]     #TODO: precompute in final algorithm for material space triangles
	A =  np.cross(d1,d2)/2 			 #TODO: precompute in final algorithm for material space triangles
	return (d1, d2, A)

def get_R(trim,tris,testing=True):
	'''trim is a triangle in material/reference space that is 
	deformed to tris, which is a triangle in real space.  
	returns the 3x3 rotation matrix aligning both their area normals and their first shape vector.
	get_R assumes the deformation is continuous and did not invert the triangle being deformed.'''
	dm1 = trim[1]-trim[0]     #precompute in final algorithm
	dm2 = trim[2]-trim[0]     #precompute in final algorithm
	Am =  np.cross(dm1,dm2)/2 #precompute in final algorithm

	ds1 = tris[1]-tris[0]
	ds2 = tris[2]-tris[0]
	As =  np.cross(ds1,ds2)/2

	#Ra is a rotation_matrix that rotation_matrix rotates Ashat onto Amhat
	Ashat = As/np.linalg.norm(As)
	Amhat = Am/np.linalg.norm(Am)
	v2 = Ashat
	v1 = Amhat
	axisa  = np.cross(v1,v2)
	thetaa = np.arcsin(np.linalg.norm(axisa))
	if thetaa == 0.:
		Ra = np.eye(3)
	else:
		Ra = rotation_matrix(axisa, thetaa)
	if not np.isclose(np.abs(np.dot(np.dot(Ra, Amhat),Ashat)),1.).all(): #doesn't care if area's end up flipped
		Ra = Ra.T
	if testing:
		assert(np.isclose(np.abs(np.dot(np.dot(Ra, Amhat),Ashat)),1.).all())

	#Rb is a rotation_matrix that rotates the Ra*dm1 onto ds1 without unaligning the area vectors
	v1 = np.dot(Ra,dm1/np.linalg.norm(dm1))
	v2 = ds1/np.linalg.norm(ds1)
	axisb  = Ashat
	v1v2 = np.dot(v1,v2)
	if v1v2 >= 1.:
		Rb = np.eye(3)
	else:
		thetab = np.arccos(v1v2)
		Rb = rotation_matrix(axisb, thetab).T

	if not np.isclose(np.dot(Rb, v1),v2).all():
		Rb = Rb.T

	if testing:
		# test that Rb keeps the area vectors aligned
		assert(np.isclose(np.dot(Rb, v1),v2).all())
		assert(np.isclose(np.abs(np.dot(np.dot(Ra, Amhat),Ashat)),1.).all())

	R = Rb.dot(Ra)
	if testing:
		# test that R = Rb.dot(Ra).T rotates Amhat onto Ashat
		assert(np.isclose(np.abs(np.dot(R.dot(Amhat),Ashat)),1.).all())
		# test that R = (Rb*Ra).T rotates dm1 onto ds1
		assert(np.isclose(R.dot(dm1/np.linalg.norm(dm1)),ds1/np.linalg.norm(ds1)).all())
	return R

##########################################################################################
# Stretch and Shear Deformations
##########################################################################################

def align_triangles(trim,tris, testing=True):
	'''return parameters for aligning trim to tris.  
	coplanar positions are returned for trim before any shear deformation, 
	but contracting the first edges to match.'''
	dm1 = trim[1] - trim[0]  #precompute in final algorithm
	dm2 = trim[2] - trim[0]  #precompute in final algorithm
	Am = np.cross(dm1, dm2) / 2  #precompute in final algorithm

	ds1 = tris[1] - tris[0]
	ds2 = tris[2] - tris[0]
	As = np.cross(ds1, ds2) / 2

	#Ra is a rotation_matrix that rotation_matrix rotates Ashat onto Amhat
	Ashat = As / np.linalg.norm(As)
	Amhat = Am / np.linalg.norm(Am)

	R = get_R(trim, tris, testing=testing)
	if testing:
		assert (np.isclose(np.linalg.norm(R.dot(dm1)) / np.linalg.norm(dm1), 1.))

	#test that the local "x" axis is aligned by R
	xhat = ds1 / np.linalg.norm(ds1)

	#yhat is not needed for energy calculation, but is needed for deformation gradient calculation via outer product
	yhat = np.cross(As, xhat)
	yhat /= np.linalg.norm(yhat)

	#scale so "the first" edge matches between the two triangles
	xi1 = np.linalg.norm(ds1) / np.linalg.norm(dm1)
	
	if testing:
		# Test that the first vectors match.
		assert (np.isclose(xhat, R.dot(dm1) / np.linalg.norm(dm1)).all())
		assert (np.isclose(np.linalg.norm(xi1 * R.dot(dm1)), np.linalg.norm(ds1)))

	# project all  edges onto the first vector and the second vector
	xs1 = xhat.dot(ds1)  # full length
	ys1 = yhat.dot(ds1)  # zero
	xs2 = xhat.dot(ds2)
	ys2 = yhat.dot(ds2)

	# for each second vector, compute the orthogonal component
	xm1 = xhat.dot(xi1 * R.dot(dm1))
	ym1 = yhat.dot(xi1 * R.dot(dm1))
	xm2 = xhat.dot(xi1 * R.dot(dm2))
	ym2 = yhat.dot(xi1 * R.dot(dm2))

	if testing:
		# test that nothing's left out of plane using the pythagorean theorem
		assert (np.isclose(np.sqrt(xm1**2 + ym1**2), np.linalg.norm(xi1 * dm1)))
		assert (np.isclose(np.sqrt(xm2**2 + ym2**2), np.linalg.norm(xi1 * dm2)))
		assert (np.isclose(np.sqrt(xs1**2 + ys1**2), np.linalg.norm(ds1)))
		assert (np.isclose(np.sqrt(xs2**2 + ys2**2), np.linalg.norm(ds2)))

	# scale the triangle heights to match
	xi2 = ys2 / ym2

	#use a shear deformation from the heisenburg group to make the triangles match
	s = (xs2 - xm2) / ys2



	return xm2, xm1, ym2, ym1, xs2, xs1, ys2, ys1, xhat, yhat, xi1, xi2, s

# compute the 3D strain gradient
def make_S_3by3(xi1, xi2, s, xhat, yhat):
	'''returns the strain gradient in the global basis of the dynamical space.
	s = ( xs2 - xm1 ) / ys2
	xi2 = ys2 / ym2
	xi1 = norm(ds1)/norm(dm1)
	xm, ym have been rotated and scaled to be coplanar with ds1 and ds2.  '''
	projx = np.outer(xhat,xhat)
	projy = np.outer(yhat,yhat)
	projy_to_x = np.outer(xhat,yhat)    
	S = np.eye(3) + (xi2-1) * projy + xi2*s * projy_to_x 
	S *= xi1
	return S

# compute the 2D strain gradient
def make_S_2by2(xi1, xi2, s):
	'''returns the strain gradient in the local basis of the dynamical space.
	s = ( xs2 - xm2 ) / ys2
	xi2 = ys2 / ym2
	xi1 = norm(ds1)/norm(dm1)
	xm, ym have been rotated and scaled to be coplanar with ds1 and ds2.  '''
	return xi1 * np.array([[1., s*xi2],[0., xi2]])

##########################################################################################
# Putting it all together and testing it
##########################################################################################

#TODO: njit this!!
# collect the operations into polar decomposition of deformation gradient matrix.
def get_SR(trim,tris, printing=True, testing=False):
	R = get_R(trim,tris,testing=testing)
	retval = align_triangles(trim,tris, testing=testing)
	#TODO: make retval more compact
	xm2, xm1, ym2, ym1, xs2, xs1, ys2, ys1, xhat, yhat, xi1, xi2, s = retval
	S = make_S_3by3(xi1, xi2, s, xhat, yhat)
	# if (xi2<0) and printing:
	# xi2<0 is True if ym2<0 is True
	#     print('xi2 is negative.')
	if (ym2<0) and printing:
		S = make_S_3by3(xi1, xi2, s, xhat, yhat)
		print(f'ym2 is negative. detS is {np.linalg.det(S):.3f}, and detR is {np.linalg.det(R):.3f}.\r')
	return S, R    

# collect the operations into one matrix.  congrats!  you can now measure the deformation gradient F!      
def get_F(trim,tris, printing=True, testing=False):
	S, R = get_SR(trim,tris, printing=printing, testing=testing)
	F = S.dot(R)
	return F


# the explicit deformation map
def phi(F,X,b):
	return F.dot(X)  + b
def get_phi(trim,tris):
	F = get_F(trim,tris)
	b = tris[0] - F.dot(trim[0])
	return lambda X: phi(F,X,b)

def test_func(tris,trim):
	retval = align_triangles(trim,tris, testing=True)
	xm2, xm1, ym2, ym1, xs2, xs1, ys2, ys1, xhat, yhat, xi1, xi2, s = retval
	return True

def test_main(tid1 = 71, tid2 = 30):
	import trimesh
	mesh = trimesh.load('../Data/spherical_meshesspherical_mesh_64.stl')
	#subtract the center of mass
	mesh.vertices -= mesh.center_mass
	#normalize the mean radius to 1
	mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))
	# test the explicit deformation map for a number of triangles
	tris = mesh.triangles[tid1]
	trim = mesh.triangles[tid2]

	assert(test_func(tris,trim))

	#ultimate test for geom_func
	mtos = get_phi(trim,tris)
	trim_mapped = np.array([mtos(trim[0]),mtos(trim[1]),mtos(trim[2])])
	print('tris is')
	print(tris)
	print('trim is mapped to')
	print(trim_mapped)
	print('difference after mapping is')
	print(tris - trim_mapped)
	assert(np.isclose(tris - trim_mapped,0.).all())
	print(f'test successful for triangle {tid1} with triangle {tid2}.')
	return True
