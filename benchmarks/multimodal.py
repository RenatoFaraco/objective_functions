import numpy as np

def rastrigin(x):
    """
    Rastrigin function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

bounds_rastrigin = np.array([[-5.12, 5.12]] * 2)

def ackley(x):
    """
    Ackley function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n)

bounds_ackley = np.array([[-32.768, 32.768]] * 2)

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

bounds_three_hump_camel = np.array([[-5, 5]] * 2)

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

bounds_easom = np.array([[-10, 10]] * 2)

def eggholder(x, y):
    """
    Eggholder function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value.
    """
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs((x / 2 + y + 47) / 2)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

bounds_eggholder = np.array([[-512, 512]] * 2)

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

bounds_schaffer_n2 = np.array([[-100, 100]] * 2)

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

bounds_schaffer_n4 = np.array([[-100, 100]] * 2)

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

bounds_styblinski_tang = np.array([[-5, 5]] * 2)
