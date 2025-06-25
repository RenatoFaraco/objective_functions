import numpy as np
from benchmarks import classical as cla
from benchmarks import geometric as geo
from benchmarks import multimodal as mlt
from benchmarks import nonlinear as nln

_FUNCTIONS = {
    "beale": cla.beale,
    "booth": cla.booth,
    "matyas": cla.matyas,
    "rosenbrock": cla.rosenbrock,
    "sphere": cla.sphere,
    "ackley": mlt.ackley,
    "easom": mlt.easom,
    "eggholder": mlt.eggholder,
    "rastrigin": mlt.rastrigin,
    "schaffer_n2": mlt.schaffer_n2,
    "schaffer_n4": mlt.schaffer_n4,
    "styblinski_tang": mlt.styblinski_tang,
    "three_hump_camel": mlt.three_hump_camel,
    "mishra_bird_constrained": nln.mishra_bird_constrained,
    "rosenbrock_constrained": nln.rosenbrock_constrained,
    "rosenbrock_constrained_disk": nln.rosenbrock_constrained_disk,
    "simionescu": nln.simionescu,
    "townsend_modified": nln.townsend_modified,
    "bukin": geo.bukin,
    "cross_in_tray": geo.cross_in_tray,
    "goldstein_price": geo.goldstein_price,
    "holder_table": geo.holder_table,
    "levi": geo.levi,
}

_BOUNDS = {
    "beale": np.array([[-5.12, 5.12]] * 2),
    "booth": np.array([[-5.12, 5.12]] * 2),
    "matyas": np.array([[-10, 10]] * 2),
    "rosenbrock": np.array([[-2.12, 2.12], [-3, -1]]),
    "sphere": np.array([[-5.12, 5.12]] * 2),
    "ackley": np.array([[-32.768, 32.768]] * 2),
    "easom": np.array([[-10, 10]] * 2),
    "eggholder": np.array([[-512, 512]] * 2),
    "rastrigin": np.array([[-5.12, 5.12]] * 2),
    "schaffer_n2": np.array([[-100, 100]] * 2),
    "schaffer_n4": np.array([[-100, 100]] * 2),
    "styblinski_tang": np.array([[-5, 5]] * 2),
    "three_hump_camel": np.array([[-5, 5]] * 2),
    "mishra_bird_constrained": np.array([[-10, 0], [-6.5, 0]]),
    "rosenbrock_constrained": np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    "rosenbrock_constrained_disk": np.array([[-2, 2], [-2, 2]]),
    "simionescu": np.array([[-2, 2], [-2, 2]]),
    "townsend_modified": np.array([[-10, 10], [-10, 10]]),
    "bukin": np.array([[-15, -5], [-3, 3]]),
    "cross_in_tray": np.array([[-10, 10]] * 2),
    "goldstein_price": np.array([[-5.12, 5.12]] * 2),
    "holder_table": np.array([[-10, 10]] * 2),
    "levi": np.array([[-10, 10]] * 2),
}

_RESULTS = {
    "beale": 0.0,
    "booth": 0.0,
    "matyas": 0.0,
    "rosenbrock": 0.0,  # Para 2D é 0
    "sphere": 0.0,
    "ackley": 0.0,
    "easom": -1.0,
    "eggholder": -959.6407,
    "rastrigin": 0.0,
    "schaffer_n2": 0.0,
    "schaffer_n4": 0.292579,
    "styblinski_tang": -78.332,  # para 2D
    "three_hump_camel": 0.0,
    "mishra_bird_constrained": -106.7645367,
    "rosenbrock_constrained": 0.0,
    "rosenbrock_constrained_disk": 0.0,
    "simionescu": 0.0,
    "townsend_modified": 2.0052938,
    "bukin": 0.0,
    "cross_in_tray": -2.06261,
    "goldstein_price": 3.0,
    "holder_table": -19.2085,
    "levi": 0.0,
}


class BenchmarkFunction:
    def __init__(self, name: str):
        if name not in _FUNCTIONS or name not in _BOUNDS:
            raise ValueError(f"Function '{name}' not found in registry.")
        self.name = name
        self.func = _FUNCTIONS[name]
        self.bounds = _BOUNDS[name]

    def __call__(self, x):
        return self.func(x)

    def __repr__(self):
        return f"<BenchmarkFunction name={self.name}>"


FUNCTIONS = _FUNCTIONS
BOUNDS = _BOUNDS
RESULTS = _RESULTS
