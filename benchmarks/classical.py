import numpy as np

def sphere(x):
    """
    Sphere function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    x = np.asarray(x)
    return np.sum(x**2)

bounds_sphere = np.array([[-5.12, 5.12]] * 2)

def rosenbrock(x):
    """
    Rosenbrock function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

bounds_rosenbrock = np.array([[-2.12, 2.12], [-1, -3]])

def beale(x, y):
    """
    Beale function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    t1 = (1.5 - x + x * y)**2
    t2 = (2.25 - x + x * y**2)**2
    t3 = (2.625 - x + x * y**3)**2
    return t1 + t2 + t3

bounds_beale = np.array([[-5.12, 5.12]] * 2)

def booth(x, y):
    """
    Booth function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

bounds_booth = np.array([[-5.12, 5.12]] * 2)

def matyas(x, y):
    """
    Matyas function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

bounds_matyas = np.array([[-10, 10]] * 2)
