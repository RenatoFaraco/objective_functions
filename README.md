# Objective Functions in Python
## Overview

This repository provides a collection of benchmark objective functions implemented in Python, commonly used to test and compare optimization algorithms. The functions include both unconstrained and constrained continuous optimization problems, featuring various characteristics such as multimodality, valleys, plateaus, and discontinuities.

## Key Features
- **Pure Python + NumPy**: All functions leverage NumPy for vectorized computations and performance.
- **Consistent Interface**: All objective functions now accept a single NumPy array `x` of shape `(n,)`, where `n` is the number of dimensions (e.g., `x = np.array([x1, x2])`).
- **Rich Documentation**: Each function includes clear docstrings detailing the mathematical formula, parameters, and return values.
- **Predefined Search Domains**: Built‑in bounds for each function facilitate immediate use in optimization experiments.
- **Unconstrained & Constrained Variants**: Includes both standard continuous functions and constrained versions (e.g., Rosenbrock with cube, line, or disk constraints, Mishra’s Bird, Modified Townsend, Simionescu).


## Included Functions
### Classical Unconstrained Optimizers
- **Rastrigin**
- **Ackley**
- **Sphere**
- **Rosenbrock**
- **Beale**
- **Goldstein–Price**
- **Booth**
- **Bukin N.6**
- **Matyas**
- **Lévi N.13**
- **Himmelblau**
- **Three‑Hump Camel**
- **Easom**
- **Cross‑in‑Tray**
- **Eggholder**
- **Hölder Table**
- **McCormick**
- **Schaffer N.2 & N.4**
- **Styblinski–Tang**

### Constrained Optimization Variants
- **Rosenbrock** with cube & line constraints
- **Rosenbrock** with disk constraint
- **Mishra’s Bird** (constrained)
- **Modified Townsend**
- **Simionescu**

## Installation

This project uses [PDM](https://pdm.fming.dev/latest/) to manage dependencies and the virtual environment.

### Prerequisites

- Python 3.10
- PDM (install with `pip install pdm`)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/RenatoFaraco/objective_functions.git
   ```

2. **Change directory**
   ```bash
   cd objective_functions
   ```

3. **Install dependencies and create environment**
   ```bash
   pdm install
   ```

4. **Run a sample script**
   ```bash
   pdm run plot
   ```

## Usage examples

### Importinhg a function

   ```python
   from benchmarks.multimodal import ackley

   value = ackley([0.5, -0.3])
   print(f"Ackley(0.5, -0.3) = {value}")
   ```

### Using `BenchmarkFunction`

You can use the `BenchmarkFunction` class to easily access both the objective function and its bounds:

   ```python
   from benchmarks.functions_registry import BenchmarkFunction

   f = BenchmarkFunction("eggholder")
   print(f"Function name: {f.name}")
   print(f"Bounds:\n{f.bounds}")
   print(f"Value at (0, 0): {f([0, 0])}")
   ```

#### Visualizing a function (Eggholder)

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from benchmarks.functions_registry import BenchmarkFunction

   f = BenchmarkFunction("eggholder")

   x_vals = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)
   y_vals = np.linspace(f.bounds[1, 0], f.bounds[1, 1], 100)
   X, Y = np.meshgrid(x_vals, y_vals)
   Z = np.array([[f([x, y]) for x in x_vals] for y in y_vals])

   fig = plt.figure(figsize=(12, 6))

   # 3D Surface Plot
   ax3d = fig.add_subplot(121, projection='3d')
   ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
   ax3d.contour(X, Y, Z, levels=50, colors='lightgrey', offset=np.min(Z))
   ax3d.set_title('3D Surface Plot')
   ax3d.set_xlabel('X')
   ax3d.set_ylabel('Y')
   ax3d.set_zlabel('Z')

   # 2D Contour Plot
   ax2d = fig.add_subplot(122)
   contour = ax2d.contourf(X, Y, Z, levels=50, cmap='viridis')
   plt.colorbar(contour, ax=ax2d)
   ax2d.contour(X, Y, Z, colors='lightgrey', levels=10)
   ax2d.set_title('2D Contour Plot')
   ax2d.set_xlabel('X')
   ax2d.set_ylabel('Y')

   plt.tight_layout()
   plt.show()

   ```

### Importinhg a function

See the `examples/plot_function.py` script for a full demonstration using Matplotlib

## Automated testing

This project includes automated tests using `pytest`, helping ensure correctness and robustness of all benchmark functions.

### Requirements

Make sure all dependencies are installed. If you're using [PDM](https://pdm.fming.dev/latest/):

### Running the tests

To run all automated tests

   ```bash
   pdm run pytest
   ```
### Test structure

All tests are located in the `tests/` directory:

   ```bash
   tests/
   ├── test_plotting_feasibility.py
   ```
These tests verify that each benchmark function can be safely evaluated over its domain (bounds), returning finite values suitable for visualization and optimization.

## Pre-commit Hoooks

This repository uses [`pre-commit`](https://pre-commit.com) to ensure code quality and test integrity before every commit.

### Configured Hooks

- Black – Code formatting
- Flake8 – Linting
- gitlint

### Setup
```bash
pip install pre-commit
pre-commit install
```

Hooks will now automatically run on every commit.

## Contributing

Contributions are welcome!

## License

This project is licensed under the GPL‑3.0 License
