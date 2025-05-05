# Objective Functions in Python
## Overview

This repository provides a collection of benchmark objective functions implemented in Python, commonly used to test and compare optimization algorithms. The functions include both unconstrained and constrained continuous optimization problems, featuring various characteristics such as multimodality, valleys, plateaus, and discontinuities.

## Key Features  
- **Pure Python + NumPy**: All functions leverage NumPy for vectorized computations and performance :contentReference[oaicite:1]{index=1}.  
- **Rich Documentation**: Each function includes clear docstrings detailing the mathematical formula, parameters, and return values :contentReference[oaicite:2]{index=2}.  
- **Predefined Search Domains**: Built‑in bounds for each function facilitate immediate use in optimization experiments :contentReference[oaicite:3]{index=3}.  
- **Unconstrained & Constrained Variants**: Includes both standard continuous functions and constrained versions (e.g., Rosenbrock with cube, line, or disk constraints, Mishra’s Bird, Modified Townsend, Simionescu) :contentReference[oaicite:4]{index=4}.  

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
- **Styblinski–Tang** :contentReference[oaicite:5]{index=5}.  

### Constrained Optimization Variants  
- **Rosenbrock** with cube & line constraints  
- **Rosenbrock** with disk constraint  
- **Mishra’s Bird** (constrained)  
- **Modified Townsend**  
- **Simionescu** :contentReference[oaicite:6]{index=6}.  

## Installation  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/RenatoFaraco/objective_functions.git
   ```

2. **Change directory**
   ```bash
   cd objective_functions
   ```

3. **Set up a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash 
   pip install numpy pandas matplotlib seaborn
   ``` :contentReference[oaicite:7]{index=7}.
   ```

## Usage examples

### Importinhg a function

   ```python 
   from benchmarks.multimodal import ackley

   value = ackley([0.5, -0.3])
   print(f"Ackley(0.5, -0.3) = {value}")
   ```

### Importinhg a function

See the `examples/plot_function.py` script for a full demonstration using Matplotlib and Seaborn

## Contributing

Contributions are welcome! 

## License

This project is licensed under the GPL‑3.0 License