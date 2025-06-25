import numpy as np


def mishra_bird_constrained(x: np.ndarray) -> float:
    """
    Mishra's Bird function with constraints.

      Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        numpy array or float: Function value(s)
    """
    x1, x2 = x[0], x[1]
    term1 = np.sin(x2) * np.exp((1 - np.cos(x1)) ** 2)
    term2 = np.cos(x1) * np.exp((1 - np.sin(x2)) ** 2)
    term3 = (x1 - x2) ** 2
    return term1 + term2 + term3


def rosenbrock_constrained(x: np.ndarray) -> float:
    """
    Rosenbrock function with cube and line constraints.

     Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        numpy array: Function value (inf if constraints violated)
    """
    x1, x2 = x[0], x[1]
    outside_cube = (x1 < -1 or x1 > 1) or (x2 < -1 or x2 > 1)
    outside_line = x2 < -x1 + 1
    if outside_cube or outside_line:
        return 1e6
    return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


def rosenbrock_constrained_disk(x: np.ndarray) -> float:
    """
    Rosenbrock function constrained to a disk.

      Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        numpy array: Function value (inf if outside disk)
    """
    x1, x2 = x[0], x[1]
    radius = 1.5
    inside_disk = x1**2 + x2**2 <= radius**2
    Z = np.where(inside_disk, (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2, 1e6)

    return Z


def simionescu(x: np.ndarray) -> float:
    """
    Simionescu's piecewise constrained function.

       Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        numpy array or float: Function value(s)
    """
    x1, x2 = x[0], x[1]
    result = np.zeros_like(x1)
    masks = [
        (
            (-1 <= x1) & (x1 <= 0) & (-1 <= x2) & (x2 <= 0),
            lambda x1, x2: 5 * (x1 + 1) ** 2 + 5 * (x2 + 1) ** 2,
        ),
        (
            (0 <= x1) & (x1 <= 1) & (-1 <= x2) & (x2 <= 0),
            lambda x1, x2: 3 * (x1 - 1) ** 2 + 5 * (x2 + 1) ** 2,
        ),
        (
            (-1 <= x1) & (x1 <= 0) & (0 <= x2) & (x2 <= 1),
            lambda x1, x2: 5 * (x1 + 1) ** 2 + 3 * (x2 - 1) ** 2,
        ),
        (
            (0 <= x1) & (x1 <= 1) & (0 <= x2) & (x2 <= 1),
            lambda x1, x2: 3 * (x1 - 1) ** 2 + 3 * (x2 - 1) ** 2,
        ),
        (
            (-2 <= x1) & (x1 <= -1) & (-2 <= x2) & (x2 <= -1),
            lambda x1, x2: (x1 + 2) ** 2 + 5 * (x2 + 2) ** 2,
        ),
        (
            (-2 <= x1) & (x1 <= -1) & (1 <= x2) & (x2 <= 2),
            lambda x1, x2: (x1 + 2) ** 2 + 5 * (x2 - 2) ** 2,
        ),
        (
            (1 <= x1) & (x1 <= 2) & (-2 <= x2) & (x2 <= -1),
            lambda x1, x2: (x1 - 2) ** 2 + 5 * (x2 + 2) ** 2,
        ),
        (
            (1 <= x1) & (x1 <= 2) & (1 <= x2) & (x2 <= 2),
            lambda x1, x2: (x1 - 2) ** 2 + 5 * (x2 - 2) ** 2,
        ),
        (
            (-2 <= x1) & (x1 <= -1) & (-1 <= x2) & (x2 <= 1),
            lambda x1, x2: (x1 + 2) ** 2 + 3 * x2**2,
        ),
        (
            (1 <= x1) & (x1 <= 2) & (-1 <= x2) & (x2 <= 1),
            lambda x1, x2: (x1 - 2) ** 2 + 3 * x2**2,
        ),
        (
            (-1 <= x1) & (x1 <= 1) & (-2 <= x2) & (x2 <= -1),
            lambda x1, x2: 3 * x1**2 + (x2 + 2) ** 2,
        ),
        (
            (-1 <= x1) & (x1 <= 1) & (1 <= x2) & (x2 <= 2),
            lambda x1, x2: 3 * x1**2 + (x2 - 2) ** 2,
        ),
    ]

    for mask, func in masks:
        result[mask] = func(x1[mask], x2[mask])

    return result


def townsend_modified(x: np.ndarray) -> float:
    """
    Modified Townsend function for optimization.

       Parameters:
        x (np.ndarray): Vector of shape (2,), where x[0] is x1 and x[1] is x2.

    Returns:
        float or numpy array: Function value(s)
    """
    x1, x2 = x[0], x[1]
    a, b, c, d = 1.8, 1.8, 10, 10
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        result = np.empty_like(x1)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                result[i, j] = (
                    0.5 * ((x1[i, j] - a) ** 2 + (x2[i, j] - b) ** 2)
                    - np.cos(c * (x1[i, j] - a)) * np.cos(d * (x2[i, j] - b))
                    + 1
                )
        return result
    return (
        0.5 * ((x1 - a) ** 2 + (x2 - b) ** 2)
        - np.cos(c * (x1 - a)) * np.cos(d * (x2 - b))
        + 1
    )
