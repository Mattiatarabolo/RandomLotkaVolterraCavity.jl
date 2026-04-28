# RandomLotkaVolterraCavity

*Small-coupling cavity approximation for the random Generalized Lotka-Volterra system*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mattiatarabolo.github.io/RandomLotkaVolterraCavity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mattiatarabolo.github.io/RandomLotkaVolterraCavity.jl/dev/)
[![Build Status](https://github.com/Mattiatarabolo/RandomLotkaVolterraCavity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mattiatarabolo/RandomLotkaVolterraCavity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Mattiatarabolo/RandomLotkaVolterraCavity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattiatarabolo/RandomLotkaVolterraCavity.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# Overview
_RandomLotkaVolterraCavity.jl_ provides a Julia implementation of the small-coupling cavity approximation for the random **generalized Lotka-Volterra (gLV) model**.

Based on the **Gaussian Expansion Cavity Method (GECaM)**, the package computes fixed-point solutions across two regimes:
- **Finite Systems**: Individual realizations with a specific interaction structure.
- **Thermodynamic Limit**: Ensemble averages over disordered interactions.

---

## Core Types

The package uses a type-dispatched architecture to handle both finite instances and ensemble averages:

* **`Model`**: Represents a **single realization** of the interaction structure with a finite number of species $N$.
* **`ModelDisordered`**: Represents the **thermodynamic limit** ($N\to\infty$) via an ensemble average over the disorder.

---

## Fixed-Point Solvers

### GECaM (Primary Solver)

The main function `run_GECaM_FP` solves the fixed-point equations derived from the Gaussian Expansion Cavity Method. Its behavior depends on the input type:

* **Input `Model`**: Runs a **message-passing** algorithm to compute node-specific mean abundances $\mu$, correlations $q$, and integrated responses $\chi$.
* **Input `ModelDisordered`**: Runs a **population dynamics** algorithm to compute the steady-state distributions of the observables $\mu$, $q$ and $\chi$.

### Alternative Approximations

Functions `run_IBMF_FP` and `run_BP_FP` implement approximations derived from a local closure of the Fokker-Planck equations ([Machado et al., arXiv:2511.17499 (2025)](https://doi.org/10.48550/arXiv.2511.17499) ). Both support `Model` and `ModelDisordered` inputs:

1. **IBMF (Individual Based Mean Field)**: Neglects correlations between neighbors. In the GECaM framework, this is equivalent to setting correlations $q$ and responses $\chi$ to zero.
2. **BP (Belief Propagation)**: Accounts for correlations through a message-passing scheme but lacks a response term. This corresponds to the **Fluctuation-Dissipation (FDT)** regime where $\chi=\beta q$.

---

## Numerical Simulation

To validate the cavity approximations against ground-truth trajectories, the package provides:

* **`run_MC`**: A numerical solver for the explicit generalized Lotka-Volterra (gLV) ODE system.


# Installation

The package can be installed via the Julia package manager.

### Stable Version

To install the latest stable version from the official General Registry, run:

```julia-repl
pkg> add RandomLotkaVolterraCavity

```

Or via the `Pkg` API:

```julia
import Pkg; Pkg.add("RandomLotkaVolterraCavity")

```

### Development Version

To install the latest development version directly from the GitHub repository, run:

```julia-repl
pkg> add https://github.com/Mattiatarabolo/RandomLotkaVolterraCavity.jl

```

Or via the `Pkg` API:

```julia
import Pkg; Pkg.add(url="https://github.com/Mattiatarabolo/RandomLotkaVolterraCavity.jl")

```

# References

The package is based on the following paper:
- [M. Tarabolo, L. Dall'Asta & R. Mulet, Sparse Interactions Reshape Stability in Random Lotka-Volterra Dynamics, arXiv:2512.22563 (2025). Still under review.](https://doi.org/10.48550/arXiv.2512.22563)