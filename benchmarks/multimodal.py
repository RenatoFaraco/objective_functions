import numpy as np

def ackley(x: np.ndarray) -> float:
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


def easom(x: np.ndarray) -> float:
    """
    Easom function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)


def eggholder(x: np.ndarray) -> float:
    """
    Eggholder function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)/2)) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


def rastrigin(x: np.ndarray) -> float:
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


def schaffer_n2(x: np.ndarray) -> float:
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


def schaffer_n4(x: np.ndarray) -> float:
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


def styblinski_tang(x: np.ndarray) -> float:
    """
    Styblinski-Tang function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return 0.5 * (x1**4 - 16*x1**2 + 5*x1 + x2**4 - 16*x2**2 + 5*x2)


def three_hump_camel(x: np.ndarray) -> float:
    """
    Three-Hump Camel function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value (inf if constraints are violated)
    """
    x1, x2 = x[0], x[1]
    return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2






