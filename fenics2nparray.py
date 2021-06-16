from fenics import *
import numpy as np

def fenics2nparray_2D(data, boundary_value, x, y):
    """
    Writes fenics solution to numpy array.

    Arguments: 
        data (fenics solution): input data 
        boundary_value (array): if point from numpy array not found on fenics mesh, 
            value is set to boundary_value
        x (array): x values of numpy array
        y (array): y values of numpy array  

    Returns: 
        data_grid (array): output data
    """
    X,Y = np.meshgrid(x,y)
    Nx = len(y)
    Ny = len(x)	
    data_grid = np.zeros([Nx, Ny])

    for i in range(Nx):
        for j in range(Ny):
            x_val = X[i,j]
            y_val = Y[i,j]
            point = Point(x_val, y_val)
            try:
                data_val = data(point)
                data_grid[i,j] = data_val
            except:
                data_grid[i,j] = boundary_value

    return data_grid

def fenics2nparray_1D(data, x, y):
    """
    Writes fenics solution to numpy array.

    Arguments: 
        data (fenics solution): input data 
        x (array): x values of numpy array
        y (array): y values of numpy array  

    Returns: 
        data_array (array): output data
    """
    N = len(x)
    data_array = np.zeros(N)

    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        point = Point(x_val, y_val)
        data_val = data(point)
        data_array[i] = data_val

    return data_array
