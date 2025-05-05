# Objective Functions in Python
## Overview

This repository provides a collection of benchmark objective functions implemented in Python, commonly used to test and compare optimization algorithms. The functions include both unconstrained and constrained continuous optimization problems, featuring various characteristics such as multimodality, valleys, plateaus, and discontinuities.

## Included functions

The repository implements the following benchmark functions:

### Classical Optimization Functions

Rastrigin

Ackley

Sphere

Rosenbrock

Beale

Goldstein-Price

Booth

Bukin N.6

Matyas

Lévi N.13

Himmelblau

Three-Hump Camel

Easom

Cross-in-Tray

Eggholder

Hölder Table

McCormick

Schaffer N.2 & N.4

Styblinski-Tang

### Constrained Optimization Functions
Rosenbrock with cube and line constraints

Rosenbrock with disk constraint

Mishra's Bird (constrained)

Modified Townsend function

Simionescu function

## Key Features
Pure Python implementation with NumPy for efficiency

Detailed documentation for each function (parameters, mathematical formula, search domain)

Predefined search boundaries for each function

Vectorized implementations where applicable

Includes both constrained and unconstrained variants

## Applications

These functions are useful for:

Testing optimization algorithms (gradient descent, evolutionary algorithms, etc.)

Comparing the performance of different optimization methods

Developing and validating new optimization techniques

Educational purposes in numerical methods and optimization

## Requirements

Python 3.x

NumPy