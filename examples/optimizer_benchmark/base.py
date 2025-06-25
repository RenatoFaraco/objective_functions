from abc import ABC, abstractmethod


class BenchmarkRunner(ABC):
    def __init__(self, functions, bounds, known_results, n_runs=5):
        self.functions = functions
        self.bounds = bounds
        self.known_results = known_results
        self.n_runs = n_runs
        self.results = []

    @abstractmethod
    def run(self):
        pass
