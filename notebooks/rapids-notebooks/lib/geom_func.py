import numpy as np
# Geometric Functions for rotating/deforming/aligning triangles.  
# For the purpose of measuring deformation gradient between two triangles 



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
    v1 = Ashat
    v2 = Amhat
    axisa = np.cross(v1,v2)
    thetaa = -np.arcsin(np.dot(axisa,axisa/np.linalg.norm(axisa)))
    Ra = -rotation_matrix(axisa, thetaa)
    
    if not np.isclose(np.dot(Ra, v1),v2).all():
        Ra *= -1
        print('Ra was flipped likely because of the sign ambiguity of thetaa.')
    if testing:
        assert(np.isclose(np.dot(Ra, v1),v2).all())#v1,v2 are about to be renamed'
        
    #Rb is a rotation_matrix that rotates the Ra*dm1 onto ds1 without unaligning the area vectors
    v1 = np.dot(Ra,ds1/np.linalg.norm(ds1))
    v2 = dm1/np.linalg.norm(dm1)
    axisb  = Amhat 
    thetab = np.arccos(np.dot(v1,v2))
#     axisb = np.cross(v1,v2)
#         w = np.cross(v1,v2)
#     thetab = np.arcsin(np.linalg.norm(w))
#     thetab = -np.arcsin(np.dot(w,w/np.linalg.norm(w)))
    Rb = rotation_matrix(axisb, thetab).T

    if not np.isclose(np.dot(Rb, v1),v2).all():
        print(np.dot(Rb, v1))
        print(v2)
        print('Rb was flipped likely because of the sign ambiguity of thetab.')
        Rb = Rb.T
        print(np.dot(Rb, v1))
        print(v2)
        
    R = Rb.dot(Ra).T
    if testing:
        assert(np.isclose(np.dot(Rb, v1),v2).all())
        # test that Rb keeps the area vectors aligned
        print(Rb.dot(Ra.dot(Ashat)))
        print(Amhat)
        assert(np.isclose(Rb.dot(Ra.dot(Ashat)),Amhat).all())
        # test that R = Rb.dot(Ra).T rotates Amhat onto Ashat
        assert(np.isclose(Ashat,R.dot(Amhat)).all())
        # test that R = (Rb*Ra).T rotates dm1 onto ds1
        assert(np.isclose(R.dot(dm1/np.linalg.norm(dm1)),ds1/np.linalg.norm(ds1)).all())
    return R

def test_func(tris,trim):
	pass
# dm1 = trim[1]-trim[0]
# dm2 = trim[2]-trim[0]
# Am =  np.cross(dm1,dm2)/2

# ds1 = tris[1]-tris[0]
# ds2 = tris[2]-tris[0]
# As =  np.cross(ds1,ds2)/2
# #Ra is a rotation_matrix that rotation_matrix rotates Ashat onto Amhat
# Ashat = As/np.linalg.norm(As)
# Amhat = Am/np.linalg.norm(Am)
# v1 = Ashat
# v2 = Amhat
# axisa = np.cross(v1,v2)
# thetaa = -np.arcsin(np.dot(axisa,axisa/np.linalg.norm(axisa)))
# # thetaa = np.arcsin(np.linalg.norm(axisa))
# Ra = -rotation_matrix(axisa, thetaa)
# print(np.dot(Ra, v1))
# print(v2)
# assert(np.isclose(np.dot(Ra, v1),v2).all())
# if not np.isclose(np.dot(Ra, v1),v2).all():
#     Ra *= -1
# assert(np.isclose(np.dot(Ra, v1),v2).all())
# # test that R = Rb.dot(Ra).T rotates Amhat onto Ashat
# R = Rb.dot(Ra).T
# print('check that Amhat rotates to Ashat.')
# print(R.dot(Amhat))
# print(Ashat)
# assert(np.isclose(Ashat,R.dot(Amhat)).all())