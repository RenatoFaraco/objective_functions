import numpy as np

def sphere(x):
    """
    Sphere function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    return sum(xi**2 for xi in x)

limit_sphere = np.array([[-5.12, 5.12]]*2)

def rosenbrock(x):
    """
    Rosenbrock function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    n = len(x)
    result = 0
    for i in range(n-1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result

limit_rosenbrock = np.array([[-2.12, 2.12],[-1,-3]])

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

limit_beale = np.array([[-5.12, 5.12]]*2)

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

limit_booth = np.array([[-5.12, 5.12]] * 2)

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

limit_matyas = np.array([[-10, 10]] * 2)
