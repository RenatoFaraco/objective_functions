import numpy as np

def goldstein_price(x, y):
    """
    Goldstein-Price function for optimization.
    
    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
    
    Returns:
        float: Function value at (x, y).
    """
    term1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return term1 * term2

bounds_goldstein_price = np.array([[-5.12, 5.12]] * 2)

def bukin(x, y):
    """
    Bukin N.6 function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value at (x, y).
    """
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

bounds_bukin = np.array([[-15, -5], [-3, 3]])

def levi(x, y):
    """
    Lévi N.13 function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value at (x, y).
    """
    return (
        np.sin(3 * np.pi * x)**2 +
        (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2) +
        (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
    )

bounds_levi = np.array([[-10, 10]] * 2)

def holder_table(x, y):
    """
    Hölder Table function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value at (x, y).
    """
    term = np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))
    return -term

bounds_holder_table = np.array([[-10, 10]] * 2)

def cross_in_tray(x, y):
    """
    Cross-in-Tray function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value at (x, y).
    """
    x /= 100
    y /= 100
    r = np.sqrt(x**2 + y**2)
    arg = np.clip(np.abs(100 - r / np.pi), -100, 100)
    value = np.abs(np.sin(x) * np.sin(y) * np.exp(arg)) + 1
    return -0.0001 * value**0.1

bounds_cross_in_tray = np.array([[-10, 10]] * 2)
