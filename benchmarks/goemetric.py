import numpy as np

def goldstein_price(x: np.ndarray) -> float:
    """
    Goldstein-Price function for optimization.
    
    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.
         
    Returns:
        float: Function value at (x, y).
    """
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return term1 * term2

limit_goldstein_price = np.array([[-5.12, 5.12]] * 2)

def bukin(x: np.ndarray) -> float:
    """
    Bukin N.6 function for optimization.
    
    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.
    
    Returns:
        float: Function value at (x, y).
    """
    x1, x2 = x[0], x[1]
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)

limit_bukin = np.array([[-15, -5], [-3, 3]])

def levi(x: np.ndarray) -> float:
    """
    Lévi N.13 function for optimization.
    
    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value at (x, y).
    """
    x1, x2 = x[0], x[1]
    return (
        np.sin(3 * np.pi * x1)**2 +
        (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2) +
        (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2)
    )

limit_levi = np.array([[-10, 10]] * 2)

def holder_table(x: np.ndarray) -> float:
    """
    Hölder Table function for optimization.
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.
        
    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value at (x, y).
    """
    x1, x2 = x[0], x[1]
    term = np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)))
    return -term

limit_holder_table = np.array([[-10, 10]] * 2)

def cross_in_tray(x: np.ndarray) -> float:
    """
    Cross-in-Tray function for optimization.
     
    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value at (x, y).
    """
    x1, x2 = x[0], x[1]
    scaled_x = x1 / 100.0
    scaled_y = x2 / 100.0

    arg = np.abs(100 - np.sqrt(scaled_x**2 + scaled_y**2) / np.pi)
    clipped_arg = np.clip(arg, -100, 100)

    exp_term = np.exp(clipped_arg)
    return -0.0001 * (np.abs(np.sin(scaled_x) * np.sin(scaled_y) * exp_term) + 1)**0.11

limit_cross_in_tray = np.array([[-10, 10]] * 2)
