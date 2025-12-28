# 2D Quantum Solver

A numerical solver for bound states of an electron above a dielectric surface using DLL-FDM, shift–invert eigenvalue methods, and image-charge potentials.

## Overview

This solver calculates electron states in cylindrical coordinates (r, z) for systems like electrons near surfaces. It handles multiple potential types including point charges and image charges.

## Repository Structure

- **`schrodinger_2d/`**: Main 2D quantum solver
  - `schrodinger_2d_solver.py`: Core numerical solver
  - `ne.py`: Neon surface application 
  - `plot_results.py`: Visualization tools


## Installation

```bash
pip install numpy scipy matplotlib
```

## Usage

### Neon Surface Example

```bash
cd schrodinger_2d
python ne.py
```

This calculates electron states near a neon surface and generates:
- Complete energy spectrum for m=0 and m=1 states
- Energy level diagrams
- Wavefunction probability density plots


## Results (Neon Example)

```
Global energy spectrum (first 6 states):
State 0: E = -117.9 meV (m=0, n=0)  # Ground state
State 1: E = -110.6 meV (m=1, n=0)  # First excited
State 2: E = -104.1 meV (m=0, n=1)  # Second excited
State 3: E = -98.0 meV (m=1, n=1)   # Third excited
State 4: E = -92.5 meV (m=0, n=2)   # Fourth excited
State 5: E = -87.4 meV (m=1, n=2)   # Fifth excited
```


## References

1. **Original MATLAB implementation**: by Ziheng Zhang (WASHU)
2. **DLL-FDM Method**: [arXiv:1512.05826](https://arxiv.org/pdf/1512.05826)

