from examples.optimizer_benchmark import optimizers as op
from benchmarks import functions_registry as reg


def test_benchmark_execution_runs():
    runner = op.SciPyOptimizer(
        functions=reg.FUNCTIONS,
        bounds=reg.BOUNDS,
        known_results=reg.RESULTS,
        optimizers=["differential_evolution", "nelder_mead"],
        n_runs=1,
    )

    runner.run()

    assert isinstance(runner.results, list)
    assert len(runner.results) == 46
    for r in runner.results:
        assert "function" in r
        assert "optimizer" in r
        assert "fun" in r
        assert "nfev" in r
        assert "time" in r
        assert "success" in r
