# GBasis

[![pytest](https://github.com/theochem/gbasis/actions/workflows/pytest.yaml/badge.svg)](https://github.com/theochem/gbasis/actions/workflows/pytest.yaml)
[![PyPI](https://img.shields.io/pypi/v/qc-gbasis.svg)](https://pypi.python.org/pypi/qc-gbasis/)
![Version](https://img.shields.io/pypi/pyversions/qc-gbasis.svg)
![License](https://img.shields.io/github/license/theochem/gbasis)
<!-- [![release](https://github.com/theochem/gbasis/actions/workflows/release.yaml/badge.svg)](https://github.com/theochem/gbasis/actions/workflows/release.yaml) -->
<!-- [![CodeFactor](https://www.codefactor.io/repository/github/tovrstra/stepup-core/badge)](https://www.codefactor.io/repository/github/tovrstra/stepup-core) -->

## About

`gbasis` is a pure-Python package for evaluating and analytically integrating Gaussian-type orbitals
and their related quantities. The goal is to build a set of tools to the quantum chemistry community
that are easily accessible and easy to use as to facilitate future scientific works.

Since basis set manipulation is often slow, Quantum Chemistry packages in Python often interface to
a lower-level language, such as C++ and Fortran, for these parts, resulting in a more difficult
build process and limited distribution. The hope is that `gbasis` can fill in this gap without a
significant difference in performance.

See [the `gbasis` website](https://gbasis.qcdevs.org/) for more information, tutorials and examples,
and API documentation.

## Citation

Please use the following citation in any publication using `gbasis` library:

> **"GBasis: A Python Library for Evaluating Functions, Functionals, and Integrals Expressed with
> Gaussian Basis Functions."**,
> T. D. Kim, L. Pujal, M. Richer, M. van Zyl, M. Martínez-González, A. Tehrani, V. Chuiko,
> G. Sánchez-Díaz, W. Sanchez, W. Adams, X. Huang, B. D. Kelly, E. Vöhringer-Martinez,
> T. Verstraelen, F. Heidar-Zadeh, and P. W. Ayers, *accepted for publication*.

## Installation

[See the website for installation instructions.](https://gbasis.qcdevs.org/installation.html)

## Feature List (Partial)

This is a partial list of the features that are supported in `gbasis`:

### Importing basis set

- from Gaussian94 basis set file (`gbasis.parsers.parse_gbs`)
- from NWChem basis set file (`gbasis.parsers.parse_nwchem`)
- from `iodata` (`gbasis.wrappers.from_iodata`)
- from `pyscf` (`gbasis.wrappers.from_pyscf`)

### Evaluations

- of basis sets (`gbasis.eval.evaluate_basis`)
- of arbitrary derivative of basis sets (`gbasis.eval_deriv.evaluate_deriv_basis`)
- of density (`gbasis.density.evaluate_density`)
- of arbitrary derivative of density (`gbasis.density.evaluate_deriv_density`)
- of gradient of density (`gbasis.density.evaluate_density_gradient`)
- of Laplacian of density (`gbasis.density.evaluate_density_laplacian`)
- of Hessian of density (`gbasis.density.evaluate_density_hessian`)
- of stress tensor (`gbasis.stress_tensor.evaluate_stress_tensor`)
- of Ehrenfest force (`gbasis.stress_tensor.evaluate_ehrenfest_force`)
- of Ehrenfest Hessian (`gbasis.stress_tensor.evaluate_ehrenfest_hessian`)
- of positive-definite kinetic energy (`gbasis.density.evaluate_posdef_kinetic_energy_density`)
- of general form of the kinetic energy (`gbasis.density.evaluate_general_kinetic_energy_density`)
- of electrostatic potential (`gbasis.electrostatic_potential.electrostatic_potential`)

### Integrals

- overlap integrals of a basis set (`gbasis.overlap.overlap_integral`)
- overlap integrals between two basis sets (`gbasis.overlap_asymm.overlap_integral_asymmetric`)
- arbitrary multipole moment integral (`gbasis.moment.moment_integral`)
- kinetic energy integral (`gbasis.kinetic_energy.kinetic_energy.integral`)
- momentum integral (`gbasis.momentum.momentum_integral`)
- angular momentum integral (`gbasis.angular_momentum.angular_momentum_integral`)
- point charge interaction integral (`gbasis.point_charge.point_charge_integral`)
- nuclear-electron attraction integral (`gbasis.point_charge.point_charge_integral`)
- electron-electron repulsion integral (`gbasis.electron_repulsion.electron_repulsion_integral`)

## Acknowledgements

This software was developed using funding from a variety of international sources including, but not
limited to: Canarie, the Canada Research Chairs, Compute Canada, the European Union's Horizon 2020
Marie Skłodowska-Curie Actions (Individual Fellowship No 800130), the Foundation of Scientific
Research--Flanders (FWO), McMaster University, the National Fund for Scientific and Technological
Development of Chile (FONDECYT), the Natural Sciences and Engineering Research Council of Canada
(NSERC), the Research Board of Ghent University (BOF), and Sharcnet.
