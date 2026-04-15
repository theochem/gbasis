"""Module for interfacing to the PySCF quantum chemistry package."""

import numpy as np
from gbasis.contractions import GeneralizedContractionShell

def from_pyscf(mol):
    """Return basis set stored within the `Mole` instance in `pyscf`.

    Parameters
    ----------
    mol : pyscf.gto.mole.Mole
        `Mole` object in `pyscf`.

    Returns
    -------
    basis : tuple of gbasis.contraciton.GeneralizedContractionShell
        Contractions for each atom.
        Contractions are ordered by the atom first, then the contractions as ordered in `pyscf`.

    Raises
    ------
    ValueError
        If `mol` is not a `pyscf.gto.mole.Mole` instance.

    Notes
    -----
    This function touches the internal components of `pyscf`, which may or may not be documented.
    This function will break as soon as the internal components change. If so, please raise an issue
    at https://github.com/theochem/gbasis. It is supported for, at least, `pyscf` version 1.6.1.

    """
    # pylint: disable=W0212
    if not (mol.__class__.__name__ == "Mole" and hasattr(mol, "_basis")):
        raise ValueError("`mol` must be a `pyscf.gto.mole.Mole` instance.")

    # assign the coordinate types (which can be either Cartesian or Spherical)
    # it seems like pyscf does not support mixed "cartesian" and "spherical" basis.
    coord_types = "cartesian" if mol.cart else "spherical"

    class PyscfShell(GeneralizedContractionShell):
        """Shell object that is compatible with gbasis' shell object.

        See `gbasis.contractions.GeneralizedContractionShell` for the documentation.

        """

        @property
        def angmom_components_sph(self):
            """Return the ordering of the magnetic quantum numbers for the given angmom in pyscf."""
            if self.angmom == 1:
                return ("c1", "s1", "c0")
            return super().angmom_components_sph

    basis = []
    for atom, coord in mol._atom:
        basis_info = mol._basis[atom]

        for shell in basis_info:
            angmom = shell[0]
            exps_coeffs = np.vstack(shell[1:])
            exps = np.array(exps_coeffs[:, 0])
            coeffs = np.array(exps_coeffs[:, 1:])
            basis.append(PyscfShell(angmom, np.array(coord), coeffs, exps, coord_types))

    return tuple(basis)