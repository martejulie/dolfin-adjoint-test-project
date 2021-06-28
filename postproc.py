import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def plot_scalar(f, ax, **kwargs):
    '''As pcolor'''
    mesh = f.function_space().mesh()
    x, y = mesh.coordinates().T
    triangulation = tri.Triangulation(x, y, mesh.cells())

    z = f.compute_vertex_values()

    mappable = ax.tripcolor(triangulation, z, **kwargs)

    return mappable, ax


def draw_mesh(f, ax, **kwargs):
    '''Of f'''
    mesh = f.function_space().mesh()
    x, y = mesh.coordinates().T
    triangulation = tri.Triangulation(x, y, mesh.cells())

    obj = ax.triplot(triangulation, **kwargs)

    return obj, ax


def draw_contours(f, ax, **kwargs):
    '''Contour lines'''
    mesh = f.function_space().mesh()
    x, y = mesh.coordinates().T
    triangulation = tri.Triangulation(x, y, mesh.cells())

    z = f.compute_vertex_values()

    mappable = ax.tricontour(triangulation, z, **kwargs)

    return mappable, ax


def draw_sampling_points(f, points, ax, s, **kwargs):
    '''Measurement points highligted via scatter'''
    values = [f(x) for x in points]
    return ax.scatter(points[:, 0], points[:, 1], c=values, s=s, **kwargs), ax


def extrema(f):
    '''Of a dolfin function'''
    values = f.vector().get_local()
    return np.min(values), np.max(values)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    mesh = df.UnitSquareMesh(32, 32)
    V = df.FunctionSpace(mesh, 'CG', 1)
    f = df.interpolate(df.Expression('(x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)', degree=1), V)

    fig, ax = plt.subplots()
    # Basic background
    mappable, ax = plot_scalar(f, ax, shading='gouraud')
    # Contours for camparison
    countours, ax = draw_contours(f, ax=ax, levels=[0.1, 0.2, 0.3], colors='magenta')
    ax.clabel(countours, inline=1, fontsize=10)

    # Indicate location of the data points
    points = np.random.rand(10, 2)
    draw_sampling_points(f, points, ax, s=40, edgecolors='black')    

    ax.set_aspect('equal')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    fig.colorbar(mappable)

    plt.show()
