from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt
np.random.seed(1)
from fenics2nparray import fenics2nparray_2D, fenics2nparray_1D
from scipy.spatial import Delaunay
import dolfin as df
import ufl

def delaunay_mesh(points_2d):
    '''Delaunay mesh from points'''
    _, gdim = points_2d.shape
    assert gdim == 2

    tri = Delaunay(points_2d)

    return build_mesh(tri.points, tri.simplices)


def build_mesh(vertices, cells):
    '''Simplex mesh from coordinates and cell-vertex connectivity'''
    nvertices, gdim = vertices.shape

    ncells, tdim_ = cells.shape
    tdim = tdim_ - 1

    mesh = df.Mesh()
    editor = df.MeshEditor()

    cell_type = {1: 'interval',
                 2: 'triangle',
                 3: 'tetrahedron'}[tdim]
    cell_type = ufl.Cell(cell_type, gdim)

    editor.open(mesh, str(cell_type), tdim, gdim)            

    editor.init_vertices(nvertices)
    editor.init_cells(ncells)

    for vi, x in enumerate(vertices):
        editor.add_vertex(vi, x)

    for ci, c in enumerate(cells):
        editor.add_cell(ci, c)
    
    editor.close()

    return mesh

def read_experimental_pO2_from_file_radial(filename):
    
    ### import data
    data = np.load(filename)
    points_2d = np.c_[data['x'], data['y']]
    
    mesh = delaunay_mesh(points_2d)
    # NOTE: for embedding data it is useful that we preserve node order
    assert np.linalg.norm(mesh.coordinates() - points_2d) < 1E-10
    # so we can just populate P1 function with
    W = FunctionSpace(mesh, 'CG', 1)    
    V = FunctionSpace(mesh, 'CG', 1)    
    d2v = dof_to_vertex_map(V)

    p = Function(V)
    p.vector().set_local(data['z'][d2v]) 
    
    ### add boundary condition
    R_ves = data['R_ves']/2
    p_ves = data['p_ves']
    xcoor = data['vessel_xcoor']
    ycoor = data['vessel_ycoor']
    
    def boundary(x, on_boundary):
        r = np.sqrt((x[0]-xcoor)**2 + (x[1]-ycoor)**2)
        b = (r <= R_ves)
        return b
    bc = DirichletBC(V, p_ves, boundary, "pointwise")
    
    return p, V, W, bc, mesh

def read_experimental_pO2_from_file(filename):

    ### import data
    data = np.load(filename)
    x = data['x']
    y = data['y']
    Nx = len(data['x'])
    Ny = len(data['y'])
    N = int(Nx*Ny)
    p_data = data['p']
    
    mesh = RectangleMesh(Point(min(x), min(y)), Point(max(x), max(y)), Nx-1, Ny-1)
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)

    d2v = dof_to_vertex_map(V)
    
    p_vector = np.reshape(p_data, (1, N))
    p = Function(V)
    p.vector()[:] = p_vector[0][d2v]
    
    ### add boundary condition
    R_ves = data['R_ves']
    p_ves = data['p_ves']
    xcoor = data['vessel_xcoor']
    ycoor = data['vessel_ycoor']
    
    def boundary(x, on_boundary):
        r = np.sqrt((x[0]-xcoor)**2 + (x[1]-ycoor)**2)
        b = (r <= R_ves)
        return b
    bc = DirichletBC(V, p_ves, boundary, "pointwise")
    
    return p, V, W, bc, mesh

def read_pO2_from_file(filename):

    ### import data
    data = np.load(filename)
    Nx = len(data['x'])
    Ny = len(data['y'])
    N = int(Nx*Ny)
    p_exact_data = data['p']
    p_noisy_data = data['p_noisy']
    
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx-1, Ny-1)
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)

    d2v = dof_to_vertex_map(V)
    
    p_exact_vector = np.reshape(p_exact_data, (1, N))
    p = Function(V)
    p.vector()[:] = p_exact_vector[0][d2v]
    
    p_noisy_vector = np.reshape(p_noisy_data, (1, N))
    p_noisy = Function(V)
    p_noisy.vector()[:] = p_noisy_vector[0][d2v]

    ### interpolate
#    mesh = Mesh("rectangular_mesh.xml")
#    mesh = refine(mesh)
#    mesh = refine(mesh)
#    mesh = refine(mesh)
#    mesh = refine(mesh)
#    
#    V = FunctionSpace(mesh, 'CG', 1)
#    W = FunctionSpace(mesh, 'CG', 1)
#    p = interpolate(p, V)
#    p_noisy = interpolate(p_noisy, V)
   
    ### add boundary condition
    R_ves = data['R_ves']
    p_ves = data['p_ves']
    
    def boundary(x, on_boundary):
        r = np.sqrt(x[0]**2 + x[1]**2)
        b = (r <= R_ves)
        return b
    bc = DirichletBC(V, p_ves, boundary, "pointwise")
    
    return p, p_noisy, V, W, bc, mesh
    
def create_synthetic_pO2_data(mesh, hole, sigma):

    R_star = 141.       # characteristic length [um]
    M_star = 1.0e-3     # charcateristic M [mmHg/um**2]

    R_ves = 6/R_star                # vessel radius
    p_ves = 80./(M_star*R_star**2)  # pO2 at vessel wall
    M = 1
    sigma = sigma/(M_star*R_star**2)    # noise

    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)

    if hole:
        def boundary(x, on_boundary):
            eps = 0.1
            r = np.sqrt(x[0]**2 + x[1]**2)
            b = ((r < R_ves+eps) and on_boundary)
            return b
        bc = DirichletBC(V, p_ves, boundary)
    else:
        def boundary(x, on_boundary):
            eps = 0.0
            r = np.sqrt(x[0]**2 + x[1]**2)
            b = (r <= R_ves+eps)
            return b
        bc = DirichletBC(V, p_ves, boundary, "pointwise")

    ### Solve the noiseless system to find the true p
    p = TrialFunction(V)
    v = TestFunction(V)
    M = Constant(M)
    a = inner(grad(p), grad(v))*dx
    L = -M*v*dx
    p = Function(V)
    solve(a == L, p, bc)

    ### Create noisy system
    N = V.dim()
    noise = sigma*np.random.randn(N)
    p_noisy = p.copy(deepcopy=True)
    p_noisy.vector()[:] += noise

    return p, p_noisy, V, W, bc, p_ves, R_ves

def estimate_M(p_data, V, W, bc, alpha, data_norm=lambda x: inner(x, x)*dx):

    ### Solve forward problem (needed for moola)
    p = TrialFunction(V)
    v = TestFunction(V)
    M = Function(W, name='Control')
    a = inner(grad(p), grad(v))*dx
    L = -M*v*dx
    p = Function(V, name='State')

    print(a, L, solve)

    solve(a == L, p, bc)

    ### Set up the functional:
    control = Control(M)
    functional = 0.5*data_norm(p_data-p) + (alpha/2)*inner(grad(M), grad(M))*dx
    J = assemble(functional)
    rf = ReducedFunctional(J, control)

    ### Solve
    M_opt = minimize(rf, options={"disp":True}, tol=1e-10)

    ### Check solution
    p_opt = TrialFunction(V)
    a = inner(grad(p_opt), grad(v))*dx
    L = -M_opt*v*dx
    p_opt = Function(V)
    solve(a == L, p_opt, bc)

    return p_opt, M_opt

if __name__ == "__main__":

#    filename = 'synthetic_data/pO2_data_sigma_1_Ld.npz'
#    p_exact, p_noisy, V, W, bc, mesh = read_pO2_from_file(filename)
    
    #filename = 'experimental_data/experimental_dataset6.npz'
    #filename = 'experimental_data/experimental_example_1a_square.npz'
    filename = 'experimental_data/xyz_example_data_a.npz'
    p_noisy, V, W, bc, mesh = read_experimental_pO2_from_file_radial(filename)
    #p_noisy, V, W, bc, mesh = read_experimental_pO2_from_file(filename)
    
#    mesh = Mesh("synthetic_mesh.xml")
#    mesh = Mesh("rectangular_mesh_w_hole.xml")
#    hole = True
#    sigma = 0
#    p_exact, p_noisy, V, W, bc, p_ves, R_ves = create_synthetic_pO2_data(mesh, hole, sigma)
    
    alpha = 1e-1
    p_opt, M_opt = estimate_M(p_noisy, V, W, bc, alpha)
   
#    e1 = errornorm(p_exact, p_noisy)
#    e2 = errornorm(p_exact, p_opt)
#    
#    print("Error in noisy signal: ", e1)
#    print("Error in restored signal: ", e2)

    ### Save solutions
    file1 = File("results/p_noisy.pvd")
    file1 << p_noisy
#    file2 = File("results/p_exact.pvd")
#    file2 << p_exact
    file3 = File("results/p_optimal.pvd")
    file3 << p_opt
    file4 = File("results/M_optimal.pvd")
    file4 << M_opt

#    data = np.load(filename)
#    x = data['x']
#    y = data['y']
###    p_exact = fenics2nparray_2D(p_exact, 0, x, y)
#    p_noisy = fenics2nparray_2D(p_noisy, 0, x, y)
#    p_opt = fenics2nparray_2D(p_opt, 0, x, y)
#    M_opt = fenics2nparray_2D(M_opt, 0, x, y)
#    
#    np.savez('results/experimental_example_1b_alpha_2_0.npz', p_noisy=p_noisy, p_opt=p_opt, M_opt=M_opt, x=x, y=y, alpha=alpha)

    data = np.load(filename)
    x = data['x']
    y = data['y']
    p_noisy = fenics2nparray_1D(p_noisy, x, y)
    p_opt = fenics2nparray_1D(p_opt, x, y)
    M_opt = fenics2nparray_1D(M_opt, x, y)
    
    np.savez('results/xyz_example_data_a_np.npz', p_noisy=p_noisy, p_opt=p_opt, M_opt=M_opt, x=x, y=y, alpha=alpha)
