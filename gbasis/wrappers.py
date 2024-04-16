"""Module for interfacing to other quantum chemistry packages."""
from gbasis.contractions import GeneralizedContractionShell
import numpy as np


def from_iodata(mol, tol=1e-20, overlap=False):
    """Return basis set stored within the `IOData` instance in `iodata`.

    Parameters
    ----------
    mol : iodata.iodata.IOData
        `IOData` instance from `iodata` module.
    tol : float
        Tolerance used in overlap screening.
    overlap : bool
        Flag for performing overlap screening between contractions.

    Returns
    -------
    basis : tuple of gbasis.contraciton.GeneralizedContractionShell
        Basis set object used within the `gbasis` module.
        `GeneralizedContractionShell` corresponds to the `Shell` object within `iodata.basis`.

    Raises
    ------
    ValueError
        If `mol` is not an `iodata.iodata.IOData` instance.
        If the primitive normalization scheme of the shells in `IOData` instance is not "L2".
    AssertionError
        If generalized contractions are found in the shells in `IOData` instance.

    Notes
    -----
    The version of the module `iodata` must be greater than 0.1.7.

    """
    if not (
        mol.__class__.__name__ == "IOData"
        and hasattr(mol, "obasis")
        and hasattr(mol.obasis, "conventions")
        and hasattr(mol.obasis, "primitive_normalization")
        and hasattr(mol.obasis, "shells")
    ):  # pragma: no cover
        raise ValueError("`mol` must be an IOData instance.")

    # GBasis can only work with segmented basis sets.
    molbasis = mol.obasis.get_segmented()

    cart_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "c"}
    sph_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "p"}

    class IODataShell(GeneralizedContractionShell):
        """Shell object that is compatible with `gbasis`' shell object.

        See `gbasis.contractions.GeneralizedContractionShell` for the documentation.

        """

        def assign_norm_cont(self):
            """Asign 1.0 as normalization constant instead of renormalizing.

            IOData does not impose a normalization convention of the contractions
            to support file formats that follow different conventions.
            """
            num_components_cart = ((self.angmom + 1) * (self.angmom + 2)) // 2
            self.norm_cont = np.ones((num_components_cart, self.coeffs.shape[1]))

        @property
        def angmom_components_cart(self):
            r"""Return the angular momentum components as ordered within the `MolecularBasis`.

            Returns
            -------
            angmom_components_cart : np.ndarray(L, 3)
                The x, y, and z components of the angular momentum vectors
                (:math:`\vec{a} = (a_x, a_y, a_z)` where :math:`a_x + a_y + a_z = \ell`).
                `L` is the number of Cartesian contracted Gaussian functions for the given
                angular momentum, i.e. :math:`(\ell + 1) * (\ell + 2) / 2`

            """
            if self.angmom not in cart_conventions:
                # GBasis needs Cartesian conventions for all angular momenta that
                # are used, even when these only appear as Spherical functions
                # in the basis set. Any convention will do when they are not
                # set by IOData.
                return super().angmom_components_cart
            return np.array(
                [(j.count("x"), j.count("y"), j.count("z")) for j in cart_conventions[self.angmom]]
            )

        @property
        def angmom_components_sph(self):
            """Return the ordering of the magnetic quantum numbers for the given angular momentum.

            Returns
            -------
            angmom_components_sph : tuple of str
                Tuple of magnetic quantum numbers of the contractions that specifies the
                ordering after transforming the contractions from the Cartesian to spherical
                coordinate system.

            Raises
            ------
            ValueError
                If convention does not support given angular momentum.

            """
            if self.angmom not in sph_conventions:
                raise ValueError(
                    "Given convention does not support spherical contractions for the angular "
                    "momentum {0}".format(self.angmom)
                )

            return tuple(sph_conventions[self.angmom])

    if molbasis.primitive_normalization != "L2":  # pragma: no cover
        raise ValueError(
            "Only L2 normalization scheme is supported in `gbasis`. Given `IOData` instance uses "
            "primitive normalization scheme, {}".format(molbasis.primitive_normalization)
        )
    if not isinstance(tol, float):
        raise TypeError("Tolerance must be provided as a float.")
    if not isinstance(overlap, bool):
        raise TypeError("Overlap must be provided as True or False.")

    basis = []
    for shell in molbasis.shells:
        # Verify that this is not a generalized contraction.
        if shell.ncon != 1:
            raise AssertionError("Generalized contraction found. The basis should be segmented.")
        # get angular momentum
        # NOTE: GeneralizedContractionShell only accepts angular momentum as an int.
        angmom = int(shell.angmoms[0])

        # pylint: disable=E1136
        basis.append(
            IODataShell(
                angmom,
                mol.atcoords[shell.icenter],
                shell.coeffs,
                shell.exponents,
                shell.kinds[0],
                icenter=shell.icenter,
                tol=tol,
                ovr_screen=overlap,
            )
        )

    return basis


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
