from tabulate import tabulate
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
)

console = Console()


def with_progress(total_tasks):
    return progress.track(range(total_tasks), description="Executando benchmarks...")


def print_step(func_name: str, optimizer_name: str):
    console.print(
        f" - Rodando função [bold yellow]{func_name}[/] com [bold yellow]{optimizer_name}[/]...",
        highlight=False,
    )


def print_results(results):
    headers = ["Function", "Optimizer", "Best value", "Runs", "Time (s)", "Result"]
    table = []

    seen = {}
    for r in results:
        key = (r["function"], r["optimizer"])
        if key not in seen or r["fun"] < seen[key]["fun"]:
            seen[key] = r

    for r in seen.values():
        table.append(
            [
                r["function"],
                r["optimizer"],
                f"{r['fun']:.6f}",
                r["nfev"],
                f"{r['time']:.4f}",
                "✔️" if r["success"] else "❌",
            ]
        )

    print("\n🔍 Resultados do Benchmark\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
