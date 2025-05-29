import numpy as np


def beale(x: np.ndarray) -> float:
    """
    Beale function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    t1 = (1.5 - x1 + x1 * x2) ** 2
    t2 = (2.25 - x1 + x1 * x2**2) ** 2
    t3 = (2.625 - x1 + x1 * x2**3) ** 2
    return t1 + t2 + t3


def booth(x: np.ndarray) -> float:
    """
    Booth function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def matyas(x: np.ndarray) -> float:
    """
    Matyas function for optimization.

    Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float: Function value.
    """
    x1, x2 = x[0], x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return result


def sphere(x: np.ndarray) -> float:
    """
    Sphere function for optimization.

    Args:
        x (array-like): Input vector.

    Returns:
        float: Function value.
    """
    return sum(xi**2 for xi in x)
