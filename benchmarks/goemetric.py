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

limit_goldstein_price = np.array([[-5.12, 5.12]] * 2)

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

limit_bukin = np.array([[-15, -5], [-3, 3]])

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

limit_levi = np.array([[-10, 10]] * 2)

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

limit_holder_table = np.array([[-10, 10]] * 2)

def cross_in_tray(x, y):
    """
    Cross-in-Tray function for optimization.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Function value at (x, y).
    """
    scaled_x = x / 100.0
    scaled_y = y / 100.0

    arg = np.abs(100 - np.sqrt(scaled_x**2 + scaled_y**2) / np.pi)
    clipped_arg = np.clip(arg, -100, 100)

    exp_term = np.exp(clipped_arg)
    return -0.0001 * (np.abs(np.sin(scaled_x) * np.sin(scaled_y) * exp_term) + 1)**0.11

limit_cross_in_tray = np.array([[-10, 10]] * 2)
