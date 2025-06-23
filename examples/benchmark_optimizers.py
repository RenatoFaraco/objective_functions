import os
import sys
import time
import numpy as np
from tabulate import tabulate
from scipy.optimize import differential_evolution, minimize


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks import functions_registry

FUNCTIONS = functions_registry.FUNCTIONS.keys()
OPTIMIZERS = ["differential_evolution", "nelder_mead"]
N_RUNS = 5


def run_optimizer(func_name, optimizer_name):

    func = functions_registry.FUNCTIONS[func_name]
    bounds = functions_registry.BOUNDS[func_name]

    lower_bounds, upper_bounds = zip(*bounds)

    results = []

    for run in range(N_RUNS):
        start_time = time.time()

        if optimizer_name == "differential_evolution":
            result = differential_evolution(
                func,
                bounds=bounds,
                strategy="best1bin",
                tol=1e-6,
                maxiter=1000,
                polish=True,
                seed=run,
            )
        elif optimizer_name == "nelder_mead":
            x0 = np.random.uniform(low=lower_bounds, high=upper_bounds)
            result = minimize(
                func, x0, method="Nelder-Mead", options={"maxiter": 1000, "fatol": 1e-6}
            )
        else:
            raise ValueError(f"Otimizador {optimizer_name} não implementado")

        end_time = time.time()

        results.append(
            {
                "fun": result.fun,
                "nfev": getattr(result, "nfev", None),
                "success": result.success,
                "time": end_time - start_time,
            }
        )

    return results


def show_results_table(all_results):
    table = []
    headers = [
        "Função",
        "Otimizador",
        "Melhor Valor",
        "Avaliações",
        "Tempo (s)",
        "Sucesso",
    ]

    grouped = {}
    for r in all_results:
        key = (r["function"], r["optimizer"])
        if key not in grouped or r["fun"] < grouped[key]["fun"]:
            grouped[key] = r

    for (func_name, optimizer_name), r in grouped.items():
        table.append(
            [
                func_name,
                optimizer_name,
                f"{r['fun']:.6f}",
                r["nfev"],
                f"{r['time']:.4f}",
                r["success"],
            ]
        )

    print("\n==== Resultados ====\n")
    print(tabulate(table, headers=headers, tablefmt="grid"))


def main():
    all_results = []

    for func_name in FUNCTIONS:
        for optimizer_name in OPTIMIZERS:
            print(f"Rodando {optimizer_name} na função {func_name}...")
            results = run_optimizer(func_name, optimizer_name)

            for r in results:
                r["function"] = func_name
                r["optimizer"] = optimizer_name
                all_results.append(r)

    show_results_table(all_results)
    print("Benchmarking concluído.")


if __name__ == "__main__":
    main()
