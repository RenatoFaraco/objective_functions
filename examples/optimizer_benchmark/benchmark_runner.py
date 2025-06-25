import os
import sys
from .optimizers import SciPyOptimizer
from .printer import print_results

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from benchmarks import functions_registry as reg


def main():
    runner = SciPyOptimizer(
        functions=reg.FUNCTIONS,
        bounds=reg.BOUNDS,
        known_results=reg.RESULTS,
        optimizers=["differential_evolution", "nelder_mead"],
        n_runs=5,
    )
    runner.run()
    print_results(runner.results)


if __name__ == "__main__":
    main()
