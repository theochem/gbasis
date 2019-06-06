# gbasis
`gbasis` is a pure-Python package for evaluating and analytically integrating Gaussian-type orbitals
and their related quantities. The goal is to build a set of tools to the quantum chemistry community
that are easily accessible and easy to use as to facilitate future scientific works.

Since basis set manipulation is often slow, Quantum Chemistry packages in Python often interface to
a lower-level lanaguage, such as C++ and Fortran, for these parts, resulting in a more difficult
build process and limited distribution. The hope is that `gbasis` can fill in this gap without a
significant difference in performance.

## Dependencies
- numpy >= 1.10
- scipy >= 1.0

## Installation
### From PyPi
Note: This is not supported yet.
```bash
pip install --user gbasis
```

### From Conda
Note: This is not supported yet.
```bash
pip install gbasis -c theochem
```

### From GitHub Repository
To install `gbasis` by itself,
```bash
git clone https://github.com/theochem/gbasis.git
cd gbasis
pip install --user -e .[dev]
```
To install `gbasis` with `pyscf`,
```bash
git clone https://github.com/theochem/gbasis.git
cd gbasis
pip install --user -e .[dev,pyscf]
```
To install `iodata`,
```bash
git clone https://github.com/theochem/iodata.git
cd iodata
pip install --user -e .
```
`iodata` must be installed separately.

To test the installation,
```bash
tox -e qa
```

## Features
Following features are supported in `gbasis`:

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
