# GBasis

[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/downloads)
[![pytest](https://github.com/theochem/gbasis/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/theochem/gbasis/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/theochem/gbasis/graph/badge.svg?token=QPUfAWj7vf)](https://codecov.io/gh/theochem/gbasis)
[![PyPI](https://img.shields.io/pypi/v/qc-gbasis.svg)](https://pypi.python.org/pypi/qc-gbasis/)
![License](https://img.shields.io/github/license/theochem/gbasis)
[![documentation](https://github.com/theochem/gbasis/actions/workflows/build_website.yaml/badge.svg?branch=master)](https://github.com/theochem/gbasis/actions/workflows/build_website.yaml)

## About

`gbasis` is a pure-Python package for analytical integration and evaluation of Gaussian-type orbitals
and their related quantities. The goal is to build a set of tools for the quantum chemistry community
that are easily accessible and extendable to facilitate future scientific works.

Since basis set manipulation is often slow, quantum chemistry packages in Python often interface to
a lower-level language, such as C++ and Fortran, resulting in a complicated build process and limited
distribution. The hope is that `gbasis` can fill in this gap without a significant difference in performance.

See [the `gbasis` website](https://gbasis.qcdevs.org/) for more information, tutorials and examples,
and API documentation.

## Citation

Please use the following citation in any publication using `gbasis` library:

> **"GBasis: A Python Library for Evaluating Functions, Functionals, and Integrals Expressed with
> Gaussian Basis Functions.\"**,
> T. D. Kim, L. Pujal, M. Richer, M. van Zyl, M. Martínez-González, A. Tehrani, V. Chuiko,
> G. Sánchez-Díaz, W. Sanchez, W. Adams, X. Huang, B. D. Kelly, E. Vöhringer-Martinez,
> T. Verstraelen, F. Heidar-Zadeh, and P. W. Ayers,
> [J. Chem. Phys. 161, 042503 (2024)](https://doi.org/10.1063/5.0216776).

## Installation

To install the latest release of `qc-gbasis`, run as follows:

```bash
python -m pip install qc-gbasis
```

See https://gbasis.qcdevs.org/installation.html for full details.

## Contributing

We welcome contributions of all kinds, such as new features,
improvements, bug fixes, and documentation clarifications. Please read
our [Contributor Guide](https://github.com/theochem/gbasis/blob/master/CONTRIBUTING.md) and
[Code of Conduct](https://github.com/theochem/.github/blob/main/CODE_OF_CONDUCT.md)
for more details.

## Feature List (Partial)

This is a partial list of the features that are supported in `gbasis`:

### Importing basis set

- from Gaussian94 basis set file (`gbasis.parsers.parse_gbs`)
- from NWChem basis set file (`gbasis.parsers.parse_nwchem`)
- from `iodata` (`gbasis.wrappers.from_iodata`)
- from `pyscf` (`gbasis.wrappers.from_pyscf`)

### Evaluations

- of basis sets (`gbasis.evals.eval.evaluate_basis`)
- of arbitrary derivative of basis sets (`gbasis.evals.eval_deriv.evaluate_deriv_basis`)
- of density (`gbasis.evals.density.evaluate_density`)
- of arbitrary derivative of density (`gbasis.evals.density.evaluate_deriv_density`)
- of gradient of density (`gbasis.evals.density.evaluate_density_gradient`)
- of Laplacian of density (`gbasis.evals.density.evaluate_density_laplacian`)
- of Hessian of density (`gbasis.evals.density.evaluate_density_hessian`)
- of stress tensor (`gbasis.evals.stress_tensor.evaluate_stress_tensor`)
- of Ehrenfest force (`gbasis.evals.stress_tensor.evaluate_ehrenfest_force`)
- of Ehrenfest Hessian (`gbasis.evals.stress_tensor.evaluate_ehrenfest_hessian`)
- of positive-definite kinetic energy (`gbasis.evals.density.evaluate_posdef_kinetic_energy_density`)
- of general form of the kinetic energy (`gbasis.evals.density.evaluate_general_kinetic_energy_density`)
- of electrostatic potential (`gbasis.evals.electrostatic_potential.electrostatic_potential`)

### Integrals

- overlap integrals of a basis set (`gbasis.integrals.overlap.overlap_integral`)
- overlap integrals between two basis sets (`gbasis.integrals.overlap_asymm.overlap_integral_asymmetric`)
- arbitrary multipole moment integral (`gbasis.integrals.moment.moment_integral`)
- kinetic energy integral (`gbasis.integrals.kinetic_energy.kinetic_energy_integral`)
- momentum integral (`gbasis.integrals.momentum.momentum_integral`)
- angular momentum integral (`gbasis.integrals.angular_momentum.angular_momentum_integral`)
- point charge interaction integral (`gbasis.integrals.point_charge.point_charge_integral`)
- nuclear-electron attraction integral (`gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_integral`)
- electron-electron repulsion integral (`gbasis.integrals.electron_repulsion.electron_repulsion_integral`)
