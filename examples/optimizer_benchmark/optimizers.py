import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from base import BenchmarkRunner
from printer import progress


class SciPyOptimizer(BenchmarkRunner):
    def __init__(self, functions, bounds, known_results, optimizers, n_runs=5):
        super().__init__(functions, bounds, known_results, n_runs)
        self.optimizers = optimizers

    def run(self):
        total = len(self.functions) * len(self.optimizers) * self.n_runs
        with progress:
            task = progress.add_task("[green]Benchmarking", total=total)

            for func_name in self.functions:
                func = self.functions[func_name]
                bounds = self.bounds[func_name]
                true_val = self.known_results[func_name]
                lower, upper = zip(*bounds)

                for optimizer_name in self.optimizers:
                    progress.update(
                        task,
                        description=f"[green]Rodando: [yellow]{func_name} + {optimizer_name}",
                    )
                    for run in range(self.n_runs):
                        start = time.time()
                        if optimizer_name == "differential_evolution":
                            result = differential_evolution(
                                func,
                                bounds=bounds,
                                strategy="best1bin",
                                tol=1e-6,
                                maxiter=1000,
                                seed=run,
                            )
                        elif optimizer_name == "nelder_mead":
                            x0 = np.random.uniform(low=lower, high=upper)
                            result = minimize(
                                func,
                                x0,
                                method="Nelder-Mead",
                                options={"maxiter": 1000, "fatol": 1e-6},
                            )
                        else:
                            raise ValueError(
                                f"Otimizador {optimizer_name} não suportado."
                            )

                        end = time.time()
                        self.results.append(
                            {
                                "function": func_name,
                                "optimizer": optimizer_name,
                                "fun": result.fun,
                                "nfev": getattr(result, "nfev", None),
                                "time": end - start,
                                "success": np.isclose(
                                    result.fun, true_val, atol=1e-6, rtol=1e-6
                                ),
                            }
                        )
                        progress.update(task, advance=1)
