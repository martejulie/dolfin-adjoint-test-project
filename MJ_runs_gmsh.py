from gmsh_interpot import msh_gmsh_model, mesh_from_gmsh
from itertools import chain
import numpy as np
import gmsh

from fenics import *
from dolfin_adjoint import *

from fenics2nparray import fenics2nparray_2D, fenics2nparray_1D

from dolfin_adjoint_test_problem import estimate_M
import dolfin as df
from fenics import *
from dolfin_adjoint import *
import sys

from gmsh_mesh import Disk, gmsh_mesh, RectangleHole


data = np.load('experimental_data/xyz_example_data_rectangular.npz')
R_ves = data['R_ves']/2.0
p_ves = data['p_ves']
x_a = data['x']
y_a = data['y']
embed_points_a = np.c_[x_a, y_a]
data = np.load('experimental_data/xyz_example_data_radial.npz')
xcoor = data['vessel_xcoor']
ycoor = data['vessel_ycoor']
x_b = data['x']
y_b = data['y']
embed_points_b = np.c_[x_b, y_b]
#ll = np.min([np.min(embed_points_a, axis=0), np.min(embed_points_b, axis=0)]) - np.array([0.01, 0.01])
ll = np.min(embed_points_a, axis=0) - np.array([0.1, 0.1])
ur = np.max([np.max(embed_points_a, axis=0), np.max(embed_points_b, axis=0)]) + np.array([0.01, 0.01])
cx, cy = xcoor, ycoor 
center = np.array([cx, cy])
radius = R_ves

if False:
    data = np.load('experimental_data/xyz_example_data_rectangular.npz')
    x = data['x']
    y = data['y']
    z = data['z']

#        cx, cy = xcoor, ycoor 
#
#        center = np.array([cx, cy])
#        radius = R_ves
    
    embed_points = np.c_[x, y]
    data_points = z
    
    # Make life easier and kick out those close to boundry to avoid intersections
    keep = np.where(np.linalg.norm(embed_points - center, 2, axis=1)**2 > (1.0001*radius)**2)
    print(len(keep[0]), len(embed_points))
    embed_points = embed_points[keep]
    data_points = data_points[keep]

    #ll = np.min(embed_points, axis=0) - np.array([0.1, 0.1])
    #ur = np.max(embed_points, axis=0) + np.array([0.1, 0.1])
#        ll = np.min(embed_points, axis=0) - np.array([0.01, 0.01])
#        ur = np.max(embed_points, axis=0) + np.array([0.01, 0.01])
#    bounding_shape = RectangleHole(ll, ur, center=center, radius=radius)
    outer_r = 1.1*np.max(np.linalg.norm(embed_points-center, 2, axis=1))
    perim = 2*(ur[0] - ll[0]) + 2*(ur[1] - ll[1])
    bounding_shape = RectangleHole(ll, ur, center=center, radius=radius,
                                  sizes={'in_min': perim/400, 'in_max': 0.025,
                                         'out_min': 2*np.pi*outer_r/200, 'out_max': 0.025})

    # NOTE: We want all the points to be strictly inside the boundary
    mesh, entity_functions, inside_points = gmsh_mesh(embed_points,
                                                      bounding_shape=bounding_shape,
                                                      argv=sys.argv)
    
    mesh_coordinates = mesh.coordinates()
    # Let's check point embedding
    vertex_function = entity_functions[0]
    vertex_values = vertex_function.array()

    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order

    # NOTE: account for points that have been kicked out
    embed_points = embed_points[inside_points]
    data_points = data_points[inside_points]

    assert np.linalg.norm(mesh_coordinates[nnz_idx] - embed_points[gmsh_idx]) < 1E-13

    # Populating a P1 function
    V = df.FunctionSpace(mesh, 'CG', 1)
    f = df.Function(V)

    v2d = df.vertex_to_dof_map(V)
    values = f.vector().get_local()
    values[v2d[nnz_idx]] = data_points[gmsh_idx]
    f.vector().set_local(values)

    assert all(abs(f(x) - target) < 1E-13 for x, target in zip(embed_points, data_points))

    # Of course, the function is incomplete
    df.File(f'results/gmsh/gmsh_foo_{bounding_shape}.pvd') << f
    # Lets check tagging 1 is circle and the rest is square boundary
    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)
    facet_f = entity_functions[1].array()
    
    inner_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 1)[0]]))
    inner_vertices = mesh_coordinates[inner_indices]
    assert np.all((np.linalg.norm(inner_vertices - center, 2, axis=1) - radius) < 1E-10), np.all((np.linalg.norm(inner_vertices - center, 2, axis=1) - radius) < 1E-10) 

    for tag in (2, 3, 4, 5):
        outer_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == tag)[0]]))
        outer_vertices = mesh_coordinates[outer_indices]
        assert np.all(
            np.logical_or(np.logical_or(np.abs(outer_vertices[:, 0]-ll[0]) < 1E-10,
                                        np.abs(outer_vertices[:, 1]-ll[1]) < 1E-10),
                          np.logical_or(np.abs(outer_vertices[:, 0]-ur[0]) < 1E-10,
                                        np.abs(outer_vertices[:, 1]-ur[1]) < 1E-10))
        )

    facet_f = entity_functions[1]

    point_f = entity_functions[0]
    point_tags = point_f.array()
    point_tags[point_tags > 0] = 1

    df.File('results/gmsh/mesh.xml') << mesh
    df.File('results/gmsh/facet_f.xml') << facet_f
    df.File('results/gmsh/point_f.xml') << point_f
    # TODO: save f in a loadable format
    with df.XDMFFile('results/gmsh/data.xdmf') as out:
        out.write_checkpoint(f, "f")

if True:
    data = np.load('experimental_data/xyz_example_data_radial.npz')
    x = data['x']
    y = data['y']
    z = data['z']
#        R_ves = data['R_ves']
#        p_ves = data['p_ves']
#        xcoor = data['vessel_xcoor']
#        ycoor = data['vessel_ycoor']
#
#        cx, cy = xcoor, ycoor 
#
#        center = np.array([cx, cy])
#        radius = R_ves
    
    embed_points = np.c_[x, y]
    data_points = z
    
    # Make life easier and kick out those close to boundry to avoid intersections
    keep = np.where(np.linalg.norm(embed_points - center, 2, axis=1)**2 > (1.0001*radius)**2)
    print(len(keep[0]), len(embed_points))
    embed_points = embed_points[keep]
    data_points = data_points[keep]
    
#    perim = 2*(ur[0] - ll[0]) + 2*(ur[1] - ll[1])
#    bounding_shape = RectangleHole(ll, ur, center=center, radius=radius,
#                              sizes={'in_min': perim/400, 'in_max': 0.025,
#                                     'out_min': 2*np.pi*outer_r/200, 'out_max': 0.025})
    #bounding_shape = RectangleHole(ll, ur, center=center, radius=radius)
    outer_r = 1.1*np.max(np.linalg.norm(embed_points-center, 2, axis=1))
    perim = 2*(ur[0] - ll[0]) + 2*(ur[1] - ll[1])
    bounding_shape = RectangleHole(ll, ur, center=center, radius=radius,
                                  sizes={'in_min': perim/400, 'in_max': 0.025,
                                         'out_min': 2*np.pi*outer_r/200, 'out_max': 0.025})

    # NOTE: We want all the points to be strictly inside the boundary
    mesh, entity_functions, inside_points = gmsh_mesh(embed_points,
                                                      bounding_shape=bounding_shape,
                                                      argv=sys.argv)
    
    mesh_coordinates = mesh.coordinates()
    # Let's check point embedding
    vertex_function = entity_functions[0]
    vertex_values = vertex_function.array()

    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order

    # NOTE: account for points that have been kicked out
    embed_points = embed_points[inside_points]
    data_points = data_points[inside_points]

    assert np.linalg.norm(mesh_coordinates[nnz_idx] - embed_points[gmsh_idx]) < 1E-13

    # Populating a P1 function
    V = df.FunctionSpace(mesh, 'CG', 1)
    f = df.Function(V)

    v2d = df.vertex_to_dof_map(V)
    values = f.vector().get_local()
    values[v2d[nnz_idx]] = data_points[gmsh_idx]
    f.vector().set_local(values)

    assert all(abs(f(x) - target) < 1E-13 for x, target in zip(embed_points, data_points))

    # Of course, the function is incomplete
    df.File(f'results/gmsh/gmsh_foo_{bounding_shape}.pvd') << f
    # Lets check tagging 1 is circle and the rest is square boundary
    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)
    facet_f = entity_functions[1].array()
    
    inner_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 1)[0]]))
    inner_vertices = mesh_coordinates[inner_indices]
    assert np.all((np.linalg.norm(inner_vertices - center, 2, axis=1) - radius) < 1E-10), np.all((np.linalg.norm(inner_vertices - center, 2, axis=1) - radius) < 1E-10) 

    for tag in (2, 3, 4, 5):
        outer_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == tag)[0]]))
        outer_vertices = mesh_coordinates[outer_indices]
        assert np.all(
            np.logical_or(np.logical_or(np.abs(outer_vertices[:, 0]-ll[0]) < 1E-10,
                                        np.abs(outer_vertices[:, 1]-ll[1]) < 1E-10),
                          np.logical_or(np.abs(outer_vertices[:, 0]-ur[0]) < 1E-10,
                                        np.abs(outer_vertices[:, 1]-ur[1]) < 1E-10))
        )

    facet_f = entity_functions[1]

    point_f = entity_functions[0]
    point_tags = point_f.array()
    point_tags[point_tags > 0] = 1


    df.File('results/gmsh/mesh.xml') << mesh
    df.File('results/gmsh/facet_f.xml') << facet_f
    df.File('results/gmsh/point_f.xml') << point_f
    # TODO: save f in a loadable format
    with df.XDMFFile('results/gmsh/data.xdmf') as out:
        out.write_checkpoint(f, "f")

mesh = Mesh('results/gmsh/mesh.xml')
facet_f = MeshFunction('size_t', mesh, 'results/gmsh/facet_f.xml')
point_f = MeshFunction('size_t', mesh, 'results/gmsh/point_f.xml')


V = FunctionSpace(mesh, 'CG', 1)
# TODO: load f from file
f = Function(V)
with df.XDMFFile('results/gmsh/data.xdmf') as datafile:
    datafile.read_checkpoint(f, "f")

dP = Measure('dP', domain=mesh, subdomain_data=point_f)

data_norm = lambda f, dP=dP: inner(f, f)*dP(1)
#data_norm = lambda f, dP=dP: inner(grad(f), grad(f))*dP(1)

p_opt, M_opt = estimate_M(p_data=f,
           V=V, 
           W=V, 
           bc=[DirichletBC(V, Constant(p_ves), facet_f, 1)],
           alpha=Constant(1),
           data_norm=data_norm)

x = mesh.coordinates()
x_data = x[point_f.array() > 0]
#print(max(abs(p_opt(xi) - f(xi)) for xi in x_data), len(x_data))

File('results/gmsh/state.pvd') << p_opt
File('results/gmsh/control.pvd') << M_opt

#    x = data['x'][keep]
#    y = data['y'][keep]
#    p_noisy = fenics2nparray_1D(f, x, y)
#    p_opt = fenics2nparray_1D(p_opt, x, y)
#    M_opt = fenics2nparray_1D(M_opt, x, y)
#    
#    np.savez('results/gmsh/results_xyz_example_data_rectangular_1000.npz', p_noisy=p_noisy, p_opt=p_opt, M_opt=M_opt, x=x, y=y) 

#x = data['x'][keep]
#y = data['y'][keep]
#p_noisy = fenics2nparray_1D(f, x, y)
##Hx = np.linspace(min(x), max(x), 100)
##Hy = np.linspace(min(y), max(y), 100)
#Hx = np.linspace(ll[0], ur[0], 100)
#Hy = np.linspace(ll[1], ur[1], 100)
#p_opt = fenics2nparray_2D(p_opt, p_ves, Hx, Hy)
#M_opt = fenics2nparray_2D(M_opt, 'NaN', Hx, Hy)
#
#np.savez('results/gmsh/results_xyz_example_data_rectangular_dense_1.npz', p_noisy=p_noisy, p_opt=p_opt, M_opt=M_opt, x=x, y=y, Hx=Hx, Hy=Hy) 
