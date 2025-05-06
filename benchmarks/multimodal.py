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

def three_hump_camel(x: np.ndarray) -> np.ndarray:
    """
    Three-Hump Camel function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value (inf if constraints are violated)
    """
    x1, x2 = x[0], x[1]
    return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2

limit_three_hump_camel = np.array([[-5, 5]] * 2)

def easom(x: np.ndarray) -> np.ndarray:
    """
    Easom function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)

limit_easom = np.array([[-10, 10]] * 2)

def eggholder(x: np.ndarray) -> np.ndarray:
    """
    Eggholder function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)/2)) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


limit_eggholder = np.array([[-512, 512]] * 2)

def schaffer_n2(x: np.ndarray) -> np.ndarray:
    """
    Schaffer N.2 function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    r = x1**2 + x2**2
    num = np.sin(np.sqrt(r))**2 - 0.5
    den = (1 + 0.001 * r)**2
    return 0.5 + num / den

limit_schaffer_n2  = np.array([[-100, 100]] * 2)

def schaffer_n4(x: np.ndarray) -> np.ndarray:
    """
    Schaffer N.4 function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    r = x1**2 + x2**2
    num = np.cos(np.sin(np.abs(x1**2 - x2**2)))**2 - 0.5
    den = (1 + 0.001 * r)**2
    return 0.5 + num / den

limit_schaffer_n4 = np.array([[-100, 100]] * 2)

def styblinski_tang(x: np.ndarray) -> np.ndarray:
    """
    Styblinski-Tang function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return 0.5 * (x1**4 - 16*x1**2 + 5*x1 + x2**4 - 16*x2**2 + 5*x2)

limit_styblinski_tang = np.array([[-5, 5]] * 2)


