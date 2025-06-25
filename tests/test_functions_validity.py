import numpy as np
import pytest
from benchmarks import functions_registry as reg  # ajuste o import se necessário


@pytest.mark.parametrize("func_name", list(reg.FUNCTIONS.keys()))
def test_function_no_nan_inf(func_name):
    func = reg.FUNCTIONS[func_name]
    bounds = reg.BOUNDS[func_name]
    lower, upper = bounds[:, 0], bounds[:, 1]
    n_samples = 100

    for _ in range(n_samples):
        x = np.random.uniform(low=lower, high=upper)
        val = func(x)
        # Verifica se val é float e se não é nan ou inf
        try:
            fval = float(val)
        except (TypeError, ValueError):
            pytest.fail(
                f"Função '{func_name}' retornou valor não numérico: {val} em x={x}"
            )
        else:
            assert not np.isnan(fval), f"Função '{func_name}' retornou NaN em x={x}"
            assert not np.isinf(fval), f"Função '{func_name}' retornou Inf em x={x}"
