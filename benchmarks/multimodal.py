import numpy as np

def rastrigin(x):
    """
    Rastrigin function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    dim = len(x)
    of = 0

    for i in range(dim):
        of += 10 + (x[i] ** 2) - 10 * np.cos(2 * np.pi * x[i])

    return of

limit_rastrigin = np.array([[-5.12, 5.12]]*2)

def ackley(x):
    """
    Ackley function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    dim = len(x)
    t1 = 0
    t2 = 0

    for i in range(dim):
        t1 += x[i] ** 2
        t2 += np.cos(2 * np.pi * x[i])

    of = 20 + np.e - 20 * np.exp((t1 / dim) * -0.2) - np.exp(t2 / dim)

    return of

limit_ackley = np.array([[-32.768, 32.768]]*2)

def three_hump_camel(x, y):
    """
    Three-Hump Camel function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

limit_three_hump_camel = np.array([[-5, 5]] * 2)

def easom(x, y):
    """
    Easom function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi)**2 - (y - np.pi)**2)

limit_easom = np.array([[-10, 10]] * 2)

def eggholder(x, y):
    """
    Eggholder function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return -(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47)/2)) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


limit_eggholder = np.array([[-512, 512]] * 2)

def schaffer_n2(x, y):
    """
    Schaffer N.2 function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    r = x**2 + y**2
    num = np.sin(np.sqrt(r))**2 - 0.5
    den = (1 + 0.001 * r)**2
    return 0.5 + num / den

limit_schaffer_n2  = np.array([[-100, 100]] * 2)

def schaffer_n4(x, y):
    """
    Schaffer N.4 function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    r = x**2 + y**2
    num = np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5
    den = (1 + 0.001 * r)**2
    return 0.5 + num / den

limit_schaffer_n4 = np.array([[-100, 100]] * 2)

def styblinski_tang(x, y):
    """
    Styblinski-Tang function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)

limit_styblinski_tang = np.array([[-5, 5]] * 2)


