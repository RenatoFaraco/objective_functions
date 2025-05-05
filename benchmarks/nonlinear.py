import numpy as np

def rosenbrock(x):
    """
    Rosenbrock function for optimization.
    
    Parameters:
        x (list or numpy array): Point coordinates in search space
        
    Returns:
        float: Function value at point x
    """
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

limit_rosenbrock = np.array([[-2.12, 2.12], [-1, -3]])

def rosenbrock_constrained(x, y):
    """
    Rosenbrock function with cube and line constraints.
    
    Parameters:
        x (numpy array): x-coordinates in search space
        y (numpy array): y-coordinates in search space
        
    Returns:
        numpy array: Function value (inf if constraints violated)
    """
    outside_cube = np.logical_or(x < -1, x > 1) | np.logical_or(y < -1, y > 1)
    outside_line = y < -x + 1
    return np.where(np.logical_or(outside_cube, outside_line), 
                  np.inf, 
                  (1-x)**2 + 100*(y-x**2)**2)

limit_rosenbrock_constrained = np.array([[-1.5, 1.5], [-0.5, 2.5]])

def rosenbrock_constrained_disk(x, y):
    """
    Rosenbrock function constrained to a disk.
    
    Parameters:
        x (numpy array): x-coordinates in search space
        y (numpy array): y-coordinates in search space
        
    Returns:
        numpy array: Function value (inf if outside disk)
    """
    return np.where(x**2 + y**2 <= 2.25, 
                  (1-x)**2 + 100*(y-x**2)**2, 
                  np.inf)

limit_rosenbrock_constrained_disk = np.array([[-2, 2], [-2, 2]])

def mishra_bird_constrained(x, y):
    """
    Mishra's Bird function with constraints.
    
    Parameters:
        x (numpy array or float): x-coordinate(s)
        y (numpy array or float): y-coordinate(s)
        
    Returns:
        numpy array or float: Function value(s)
    """
    return (np.sin(y)*np.exp((1-np.cos(x))**2) + 
            np.cos(x)*np.exp((1-np.sin(y))**2) + 
            (x-y)**2)

limit_mishra_bird_constrained = np.array([[-10, 0], [-6.5, 0]])

def townsend_modified(x, y):
    """
    Modified Townsend function for optimization.
    
    Parameters:
        x (float or numpy array): x-coordinate(s)
        y (float or numpy array): y-coordinate(s)
        
    Returns:
        float or numpy array: Function value(s)
    """
    a, b, c, d = 1.8, 1.8, 10, 10
    return 0.5*((x-a)**2 + (y-b)**2) - np.cos(c*(x-a))*np.cos(d*(y-b)) + 1

limit_townsend_modified = np.array([[-10, 10], [-10, 10]])

def simionescu(x, y):
    """
    Simionescu's piecewise constrained function.
    
    Parameters:
        x (numpy array or float): x-coordinate(s)
        y (numpy array or float): y-coordinate(s)
        
    Returns:
        numpy array or float: Function value(s)
    """
    result = np.zeros_like(x)
    masks = [
        ((-1 <= x) & (x <= 0) & (-1 <= y) & (y <= 0), lambda x,y: 5*(x+1)**2 + 5*(y+1)**2),
        ((0 <= x) & (x <= 1) & (-1 <= y) & (y <= 0), lambda x,y: 3*(x-1)**2 + 5*(y+1)**2),
        ((-1 <= x) & (x <= 0) & (0 <= y) & (y <= 1), lambda x,y: 5*(x+1)**2 + 3*(y-1)**2),
        ((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1), lambda x,y: 3*(x-1)**2 + 3*(y-1)**2),
        ((-2 <= x) & (x <= -1) & (-2 <= y) & (y <= -1), lambda x,y: (x+2)**2 + 5*(y+2)**2),
        ((-2 <= x) & (x <= -1) & (1 <= y) & (y <= 2), lambda x,y: (x+2)**2 + 5*(y-2)**2),
        ((1 <= x) & (x <= 2) & (-2 <= y) & (y <= -1), lambda x,y: (x-2)**2 + 5*(y+2)**2),
        ((1 <= x) & (x <= 2) & (1 <= y) & (y <= 2), lambda x,y: (x-2)**2 + 5*(y-2)**2),
        ((-2 <= x) & (x <= -1) & (-1 <= y) & (y <= 1), lambda x,y: (x+2)**2 + 3*y**2),
        ((1 <= x) & (x <= 2) & (-1 <= y) & (y <= 1), lambda x,y: (x-2)**2 + 3*y**2),
        ((-1 <= x) & (x <= 1) & (-2 <= y) & (y <= -1), lambda x,y: 3*x**2 + (y+2)**2),
        ((-1 <= x) & (x <= 1) & (1 <= y) & (y <= 2), lambda x,y: 3*x**2 + (y-2)**2)
    ]
    
    for mask, func in masks:
        result[mask] = func(x[mask], y[mask])
    
    return result

limit_simionescu = np.array([[-2, 2], [-2, 2]])