from gmsh_interpot import msh_gmsh_model, mesh_from_gmsh
from itertools import chain
import numpy as np
import gmsh

from fenics import *
from dolfin_adjoint import *

from fenics2nparray import fenics2nparray_1D
from gmsh_mesh import *

class Shape:
    TOL = 1E-1
    
    def is_inside(self, points):
        '''Indices of inside points'''
        pass

    def insert_gmsh(self, model, factory):
        '''Insert myself into model using factory'''
        pass

    def filter(self, truths):
        print(f'Keeping {sum(truths)}/{len(truths)}')
        return np.where(truths)


class Circle(Shape):
    '''Circle enclosing points'''
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return 'Circle'

    def is_inside(self, points):
        return self.filter(np.linalg.norm(points-self.center, 2, axis=1)**2 < self.radius**2-self.TOL)

    def insert_gmsh(self, model, factory):
        cx, cy = self.center
        circle = factory.addCircle(cx, cy, 0, self.radius)
        loop = factory.addCurveLoop([circle])
        circle = factory.addPlaneSurface([loop])

        factory.synchronize()
        
        model.addPhysicalGroup(2, [circle], tag=1)
        
        bdry = model.getBoundary([(2, circle)])
        
        for tag, curve in enumerate(bdry, 1):
            model.addPhysicalGroup(1, [curve[1]], tag)
            
        # Return the surface that embeds points
        return circle

    
class Disk(Shape):
    '''Disk enclosing points'''
    def __init__(self, center, in_radius, out_radius, sizes=None):
        assert in_radius < out_radius
        self.center = center
        self.in_radius = in_radius
        self.out_radius = out_radius
        self.sizes = sizes

    def __repr__(self):
        return 'Disk'

    def is_inside(self, points):
        return self.filter(np.logical_and(np.linalg.norm(points-self.center, 2, axis=1)**2 < self.out_radius**2-self.TOL,
                                          np.linalg.norm(points-self.center, 2, axis=1)**2 > self.in_radius**2+self.TOL))

    def insert_gmsh(self, model, factory):
        cx, cy = self.center
        
        out_circle = factory.addCircle(cx, cy, 0, self.out_radius)
        in_circle = factory.addCircle(cx, cy, 0, self.in_radius)

        factory.synchronize()

        oloop = factory.addCurveLoop([out_circle])
        iloop = factory.addCurveLoop([in_circle])        

        disk = factory.addPlaneSurface([oloop, iloop])

        factory.synchronize()
        
        model.addPhysicalGroup(2, [disk], tag=1)
        
        bdry = [p[1] for p in model.getBoundary([(2, disk)])]
        # NOTE: we have 2 surfaces marked for boundary conditions
        # The inside shall be tagged as 1 and outside as 2
        _, maybe_in = factory.getEntitiesInBoundingBox(cx-1.001*self.in_radius, cy-1.001*self.in_radius, -10,
                                                       cx+1.001*self.in_radius, cy+1.001*self.in_radius, 10, dim=1)[0]

        if bdry[0] != maybe_in:
            bdry = reversed(bdry)

        for tag, curve in enumerate(bdry, 1):
            model.addPhysicalGroup(1, [curve], tag)

        if self.sizes is not None:
            sizes = self.sizes
            dr = self.out_radius - self.in_radius
            
            fields = []
            idx = 0
            for prefix, curve in zip(('in_', 'out_'), (in_circle, out_circle)):
                idx += 1                
                model.mesh.field.add('Distance', idx)
                model.mesh.field.setNumbers(idx, 'CurvesList', [curve])
                model.mesh.field.setNumber(idx, 'NumPointsPerCurve', 100)

                idx += 1
                model.mesh.field.add('Threshold', idx)
                model.mesh.field.setNumber(idx, 'InField', idx-1)        
                model.mesh.field.setNumber(idx, 'SizeMax', sizes[f'{prefix}max'])
                model.mesh.field.setNumber(idx, 'SizeMin', sizes[f'{prefix}min'])
                model.mesh.field.setNumber(idx, 'DistMin', 0.1*dr)
                model.mesh.field.setNumber(idx, 'DistMax', 0.2*dr)

                fields.append(idx)

            idx += 1                
            model.mesh.field.add('Min', idx)
            model.mesh.field.setNumbers(idx, 'FieldsList', fields)
            model.mesh.field.setAsBackgroundMesh(idx)

            gmsh.model.occ.synchronize()
            gmsh.model.geo.synchronize()    
            
        # Return the surface that embeds points
        return disk

    
class Rectangle(Shape):
    '''Rectnangle enclosing points'''
    def __init__(self, ll, ur):
        assert np.all((ur - ll) > 0)
        self.ll = ll
        self.ur = ur

    def __repr__(self):
        return 'Rectangle'

    def is_inside(self, points):
        x, y = points.T
        return self.filter(np.logical_and(np.logical_and(x > self.ll[0]+self.TOL, x < self.ur[0]-self.TOL),
                                          np.logical_and(y > self.ll[1]+self.TOL, y < self.ur[1]-self-TOL)))

    def insert_gmsh(self, model, factory):
        ll = self.ll
        dx = self.ur - self.ll

        square = factory.addRectangle(x=ll[0], y=ll[1], z=0, dx=dx[0], dy=dx[1])

        factory.synchronize()
        bdry = model.getBoundary([(2, square)])
        model.addPhysicalGroup(2, [square], tag=1)

        for tag, curve in enumerate(bdry, 1):
            model.addPhysicalGroup(1, [curve[1]], tag)

        return square


class RectangleHole(Shape):
    '''Rectangle with circle hole enclosing points'''
    def __init__(self, ll, ur, center, radius):
        assert np.all((ur - ll) > 0)
        assert all(np.all(np.logical_and(ll < center + radius*shift, center + radius*shift < ur))
                   for shift in np.array([[1, 0],
                                          [0, 1],
                                          [-1, 0],
                                          [0, -1]]))
        self.ll = ll
        self.ur = ur
        self.center = center
        self.radius = radius

    def __repr__(self):
        return 'Rectangle'

    def is_inside(self, points):
        x, y = points.T
        return self.filter(#np.logical_and(
            # Inside square
            np.ones(len(points), dtype=bool)
            #np.logical_and(np.logical_and(x > self.ll[0]+self.TOL, x < self.ur[0]-self.TOL),
            #               np.logical_and(y > self.ll[1]+self.TOL, y < self.ur[1]-self.TOL)),
            #np.ones(len(points), dtype=bool)
            #np.linalg.norm(points - self.center, 2, axis=1)**2 > self.radius**2+self.TOL
        #)
        )
    
    def insert_gmsh(self, model, factory):
        ll = self.ll
        dx = self.ur - self.ll
        cx, cy = self.center
        radius = self.radius

        sq_points = (ll, ll+np.array([dx[0], 0]), ll+dx, ll+np.array([0, dx[1]]))
        sq_points = [factory.addPoint(x, y, z=0) for x, y in sq_points]

        n = len(sq_points)
        sq_lines = [factory.addLine(p, q) for p, q in zip(sq_points, chain(sq_points[1:], sq_points))]
        
        circle = factory.addCircle(cx, cy, 0, radius)        

        sloop = factory.addCurveLoop(sq_lines)
        cloop = factory.addCurveLoop([circle])        

        shape = factory.addPlaneSurface([cloop, sloop])

        factory.synchronize()
        
        model.addPhysicalGroup(2, [shape], tag=1)

        bdry = set([p[1] for p in model.getBoundary([(2, shape)])])
        circle_bdry,  = bdry - set(sq_lines)
        # This is circle
        model.addPhysicalGroup(1, [circle_bdry], 1)        
        # This is square
        for tag, curve in enumerate(sq_lines, 2):
            model.addPhysicalGroup(1, [curve], tag)

        return shape



# Generic
def gmsh_mesh(embed_points, bounding_shape, argv=[]):
    '''Mesh bounded by bounded shape with embedded points'''
    nembed_points, gdim = embed_points.shape

    assert gdim == 2

    gmsh.initialize(argv)

    model = gmsh.model
    factory = model.occ

    # How to bound the points returing tag of embedding surface
    bding_surface = bounding_shape.insert_gmsh(model, factory)

    inside_points, = bounding_shape.is_inside(embed_points)

    embed_points = embed_points[inside_points]
    # Embed_Points in surface we want to keep track of
    point_tags = [factory.addPoint(*point, z=0) for point in embed_points]

    factory.synchronize()    
    for phys_tag, tag in enumerate(point_tags, 1):
        model.addPhysicalGroup(0, [tag], phys_tag)
    model.mesh.embed(0, point_tags, 2, bding_surface)

    factory.synchronize()

    # NOTE: if you want to see it first
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
        
    nodes, topologies = msh_gmsh_model(model,
                                       2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions, inside_points

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin_adjoint_test_problem import estimate_M
    import dolfin as df
    from fenics import *
    from dolfin_adjoint import *
    import sys
    # NOTE: `gmsh_mesh.py -clscale 0.5`
    # will perform global refinement (halving sizes), 0.25 is even finer etc

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
    
    if True:
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
        bounding_shape = RectangleHole(ll, ur, center=center, radius=radius)

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

    if False:
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

        #ll = np.min(embed_points, axis=0) - np.array([0.1, 0.1])
        #ur = np.max(embed_points, axis=0) + np.array([0.1, 0.1])
#        ll = np.min(embed_points, axis=0) - np.array([0.01, 0.01])
#        ur = np.max(embed_points, axis=0) + np.array([0.01, 0.01])
        bounding_shape = RectangleHole(ll, ur, center=center, radius=radius)

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
               alpha=Constant(10),
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

#    from one_shot import one_shot, H10_inner
#
#    state, control, multiplier = one_shot(f, entity_functions,
#                                      state_bcs={'dirichlet': {1: Constant(p_ves)},  # 1 is inner
#                                                 'neumann': {2: Constant(0),
#                                                     3: Constant(0),
#                                                     4: Constant(0),
#                                                     5: Constant(0)}},
#                                      multiplier_bcs={'neumann': {1: Constant(0),
#                                                                  2: Constant(0),
#                                                                  3: Constant(0),
#                                                                  4: Constant(0),
#                                                                  5: Constant(0)},
#                                                      'dirichlet': {}},
#                                      alpha=Constant(1E0),
#                                      regularization=H10_inner)
#
#    File('data.pvd') << f
#    File('state.pvd') << state
#    File('control.pvd') << control
#    File('multipler.pvd') << multiplier


#
#    # Disk test
#    cx, cy = 0., 0.
#    # Synthetic
#    r = 2*np.random.rand(1000)
#    # NOTE: here we will have points with radius 2 but our domain will
#    # only have radius 1. To avoid points on the boundary we kick some
#    # out
#    tol = 0.1
#    in_radius, out_radius = 0.2, 1.0
#    
#    r = r[np.logical_and(np.logical_or(r < out_radius+tol, r > out_radius-tol),   # Outradius
#                         np.logical_or(r < in_radius+tol, r > in_radius-tol))]  # Inradius
#    th = 2*np.pi*np.random.rand(len(r))
#    x, y = cx + r*np.sin(th), cy + r*np.cos(th)
#    
#    embed_points = np.c_[x, y]
#    data_points = 2*x**2 + 3*y**2
#
#    center = np.array([cx, cy])
#    bounding_shape = Disk(center=center, out_radius=out_radius, in_radius=in_radius,
#                          sizes={'in_min': 0.01, 'in_max': 1,
#                                 'out_min': 0.1, 'out_max': 1})
#
#    # NOTE: We want all the points to be strictly inside the boundary
#    mesh, entity_functions, inside_points = gmsh_mesh(embed_points,
#                                                      bounding_shape=bounding_shape,
#                                                      argv=sys.argv)
#    
#    mesh_coordinates = mesh.coordinates()
#    # Let's check point embedding
#    vertex_function = entity_functions[0]
#    vertex_values = vertex_function.array()
#
#    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
#    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order
#
#    # NOTE: account for points that have been kicked out
#    embed_points = embed_points[inside_points]
#    data_points = data_points[inside_points]
#
#    assert np.linalg.norm(mesh_coordinates[nnz_idx] - embed_points[gmsh_idx]) < 1E-13
#
#    # Populating a P1 function
#    V = df.FunctionSpace(mesh, 'CG', 1)
#    f = df.Function(V)
#
#    v2d = df.vertex_to_dof_map(V)
#    values = f.vector().get_local()
#    values[v2d[nnz_idx]] = data_points[gmsh_idx]
#    f.vector().set_local(values)
#
#    assert all(abs(f(x) - target) < 1E-13 for x, target in zip(embed_points, data_points))
#
#    # Of course, the function is incomplete
#    df.File(f'results/gmsh_foo_{bounding_shape}.pvd') << f
#
#    # Let's also make sure that inside is labeled as 1 and outside is
#    # labeled as 2
#    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)
#    facet_f = entity_functions[1].array()
#    
#    inner_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 1)[0]]))
#    inner_vertices = mesh_coordinates[inner_indices]
#    assert np.all(np.linalg.norm(inner_vertices - center, 2, axis=1) - in_radius) < 1E-10
#
#    outer_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 2)[0]]))
#    outer_vertices = mesh_coordinates[outer_indices]
#    assert np.all(np.linalg.norm(outer_vertices - center, 2, axis=1) - out_radius) < 1E-10    
#
    # TODO:
    # - rectangle
    # - rectnagle hole
    # - one shot approach
