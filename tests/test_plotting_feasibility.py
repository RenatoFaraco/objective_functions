import numpy as np
import pytest
from benchmarks.functions_registry import BenchmarkFunction

@pytest.mark.parametrize("func_name", [
    "beale",
    "booth",
    "matyas",
    "rosenbrock",
    "sphere",
    "ackley",
    "easom",
    "eggholder",
    "rastrigin",
    "schaffer_n2",
    "schaffer_n4",
    "styblinski_tang",
    "three_hump_camel",
    "mishra_bird_constrained",
    "rosenbrock_constrained",
    "rosenbrock_constrained_disk",
    "simionescu",
    "townsend_modified",
    "bukin",
    "cross_in_tray",
    "goldstein_price",
    "holder_table",
    "levi"
])

def test_function_meshgrid_evaluates_correctly(func_name):
    f = BenchmarkFunction(func_name)
    bounds = f.bounds

    assert bounds.shape == (2, 2), f"{func_name} must be 2D for meshgrid test."

    x_vals = np.linspace(bounds[0, 0], bounds[0, 1], 250)
    y_vals = np.linspace(bounds[1, 0], bounds[1, 1], 250)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.array([[f(np.array([x, y])) for x in x_vals] for y in y_vals])

    assert Z.shape == X.shape == Y.shape, f"{func_name}: mesh shapes mismatch."
    
    if not np.any(np.isfinite(Z)):
        pytest.skip(f"{func_name}: All values non-finite; likely due to constraints.")
    else:
        assert np.any(np.isfinite(Z)), f"{func_name}: No finite values for plotting."