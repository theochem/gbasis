<!-- #region -->
# Welcome to GBasis's Documentation!

[GBasis](https://github.com/theochem/gbasis/) is a free, open-source, and cross-platform Python library designed to help you effortlessly work with Gaussian-type orbitals. Please use the following citation in any publication using the GBasis library:

> **GBasis: A Python Library for Evaluating Functions, Functionals, and Integrals Expressed with Gaussian Basis Functions.**  
> Taewon David Kim, Leila Pujal, Michelle Richer, Maximilian van Zyl, Marco Martínez-González, Alireza Tehrani, Valerii Chuiko, Gabriela Sánchez-Díaz, Wesley Sanchez, William Adams, Xiaomin Huang, Braden D. Kelly, Esteban Vöhringer-Martinez, Toon Verstraelen, Farnaz Heidar-Zadeh, Paul W. Ayers.  
> *J. Chem. Phys. 161 (4), 042503 (2024).*  
> https://doi.org/10.1063/5.0216776


```bibtex
@article{Kim2024GBasis,
  author  = {Kim, Taewon David and Pujal, Leila and Richer, Michelle and
             van Zyl, Maximilian and Mart{\'\i}nez-Gonz{\'a}lez, Marco and
             Tehrani, Alireza and Chuiko, Valerii and
             S{\'a}nchez-D{\'\i}az, Gabriela and Sanchez, Wesley and
             Adams, William and Huang, Xiaomin and Kelly, Braden D. and
             V{\"o}hringer-Martinez, Esteban and Verstraelen, Toon and
             Heidar-Zadeh, Farnaz and Ayers, Paul W.},
  title   = {{GBasis}: A Python library for evaluating functions, functionals,
             and integrals expressed with Gaussian basis functions},
  journal = {The Journal of Chemical Physics},
  volume  = {161},
  number  = {4},
  pages   = {042503},
  year    = {2024},
  doi     = {10.1063/5.0216776}
}
```

The GBasis source code is hosted on [GitHub](https://github.com/theochem/gbasis/) and is released under the [GNU LESSER GENERAL PUBLIC LICENSE](https://github.com/theochem/gbasis/blob/master/LICENSE). We welcome any contributions to the GBasis library in accordance with our [Code of Conduct](https://github.com/theochem/gbasis/blob/master/CODE_OF_CONDUCT.md); please see our [Contributing Guidelines](https://github.com/theochem/gbasis/blob/master/CONTRIBUTING.md). Please report any issues you encounter while using GBasis library on [GitHub Issues](https://github.com/theochem/gbasis/issues). For further information and inquiries please contact us at qcdevs@gmail.com.

## Why GBasis?
GBasis is a pure-Python package for evaluating and analytically integrating Gaussian-type orbitals and their related quantities. The goal is to build a set of tools to the quantum chemistry community that are easily accessible and easy to use as to facilitate future scientific works.

Since basis set manipulation is often slow, Quantum Chemistry packages in Python often interface to a lower-level lanaguage, such as C++ and Fortran, for these parts, resulting in a more difficult build process and limited distribution. The hope is that gbasis can fill in this gap without a significant difference in performance.

<!-- #endregion -->
