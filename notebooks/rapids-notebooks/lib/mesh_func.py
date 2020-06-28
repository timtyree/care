#!/bin/env python3
import trimesh
import pyglet
import numpy as np

def save_image(mesh,save_file_name='Data/render.png'):
    '''save a view of the mesh as a png'''
    scene = mesh.scene()
    png = scene.save_image(resolution=[1920, 1080], visible=True)
    print('rendered bytes:', len(png))
    with open(save_file_name, 'wb') as f:
        f.write(png)

def get_boundary(mesh):
    '''returns the boundary of trimesh.Trimesh mesh.
    Returns trimesh.path.path.Path3D bndries.
    Boundary is not guaranteed to be closed or have all
    the edges it should have.  Cleaning is necessary afterwords.'''
    faces_on_boundary = trimesh.repair.broken_faces(mesh)
    boundaries = mesh.outline(faces_on_boundary)
    b = boundaries.copy()

    # compute the number of edges attached to each vertex in b
    b.explode()
    bg = b.vertex_graph.degree

    # select only the vertices that have four edges.
    # lst is a list of vertices on the boundary of the mesh
    lst = []
    for n, item in enumerate(dict(bg)):
        if (bg[item]) >= 4:
            lst.append(item)

    bg = b.vertex_graph

    # get a list of edges in the boundaries
    # if an edge contains a point not in lst, then remove it
    # show the remaining edges

    # get the set of broken faces
    b_lst = []
    for n, entity in enumerate(b.entities):
        keep_entity = np.array([x in lst for x in entity.end_points]).all()
        if keep_entity:
            b_lst.append(entity)

    bndries = b.copy()
    bndries.entities = np.array(b_lst)
    return bndries

def post_process_boundary_1(mesh, bndries):
    '''return lst2, ofv_lst, choice_lst
    computes a list of options for patching the rough boundary returned by get_boundary.'''
    # select only the vertices that have one edges.
    bg = bndries.vertex_graph.degree

    # list is a list of stubs vertices on the boundary
    lst2 = []
    for n, item in enumerate(dict(bg)):
        if (bg[item]) == 1:
            lst2.append(item)

    ofv_lst = []
    choice_lst = []
    # for each vertex in lst, return the nearest vertex that is not in bndries already.   Then, add it
    for v in lst2:
        # boundaries.kdtree.query(mesh.vertices[v].tolist(),k=4)
        options_for_v = mesh.kdtree.query(mesh.vertices[v].tolist(), k=4)[1]
        qq = bndries.vertex_graph.nodes
        choice = 0
        choice_lst.append(choice)
        # amongst the k=four nearest neighbors of v, these are the ones that aren't already in bndries, they're listed in order or closeness to v
        ofv = options_for_v[[not vv in set(qq)
                             for vv in options_for_v]]  # [choice]

        # do this by adding the set of all relevant ofv to the lst above
        ofv_lst.append(ofv.tolist())
    return ofv_lst, choice_lst

def post_process_boundary_2(mesh, ofv_lst, choice_lst):
    '''return bndries2, which is an attempted patching of the
    result of get_boundaries.  fiddle with choice_lst until
    the result looks right with bndries.show().
    TODO: automate the fiddling of choice_lst by choosing the
    values with the smallest angle or choosing the values that
    produce no 3-edge cycles.'''
    faces_on_boundary = trimesh.repair.broken_faces(mesh)
    boundaries = mesh.outline(faces_on_boundary)
    b = boundaries.copy()
    b.explode()
    # select only the vertices that have four edges.
    bg = b.vertex_graph.degree

    # list is a list of vertices on the boundary of the mesh
    lst = []
    for n, item in enumerate(dict(bg)):
        if (bg[item]) >= 4:
            lst.append(item)

    # add an edge from ofv to lst
    for n, choice in enumerate(choice_lst):
        lst.append(ofv_lst[n][choice])

    c_lst = []
    for n, entity in enumerate(b.entities):
        keep_entity = np.array([x in lst for x in entity.end_points]).all()
        if keep_entity:
            c_lst.append(entity)
    bndries2 = b.copy()
    bndries2.entities = np.array(c_lst)
    return bndries2

def make_cap(points, attributes={'static':1, 'group_id':1}):
    '''static:1 means static=True'''
    mean = np.mean(points, axis=0) # points is a closed path
    vert = list(points)
    vert.append(mean)# the last entry is the middle of the circle
    l = len(points) #index of the mean

    #make triangles from one point to the next point with the center next
    triangles = [[i,(i+1)%l,l] for i in range(l)]

    mesh_out = trimesh.Trimesh(vertices=vert,
                           faces=triangles,
                              face_attributes= attributes)#[face_attributes for i in triangles])#,
#                               vertex_attributes= [vertex_attributes for i in triangles])
    mesh_out.fix_normals()#makes winding consistent
    return mesh_out

def merge_mesh(mesh, other):
    '''clobbers attributes of mesh and other, both of which are of type trimesh..base.Trimesh.'''
    lmv = mesh.vertices.tolist()
    lmf = mesh.faces.tolist()
    c0v = other.vertices.tolist()
    c0f = other.faces.tolist()
    c0f = (np.array(c0f)+len(lmv)).tolist()
    lmv.extend(c0v)
    lmf.extend(c0f)
    mesh_out = trimesh.Trimesh(vertices=lmv,faces=lmf)
    mesh_out.fix_normals()
    return mesh_out
