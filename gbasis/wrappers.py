"""Module for interfacing to other quantum chemistry packages."""
from gbasis.contractions import GeneralizedContractionShell
import numpy as np


def from_iodata(mol):
    """Return basis set stored within the IOData instance in iodata.

    Parameters
    ----------
    mol : iodata.iodata.IOData
        IOData instance from iodata module.

    Returns
    -------
    basis : tuple of gbasis.contraciton.GeneralizedContractionShell
        Basis set object used within the gbasis module.
        GeneralizedContractionShell corresponds to the Shell object within iodata.basis.

    Raises
    ------
    NotImplementedError
        If any contractions in the given IOData instance is spherical.
    ValueError
        If `mol` is not an iodata.iodata.IOData instance.
        If the primitive normalization scheme of the shells in IOData instance is not "L2".

    Notes
    -----
    The version of the module iodata must be greater than 0.1.7.

    """
    if not (
        mol.__class__.__name__ == "IOData"
        and hasattr(mol, "obasis")
        and hasattr(mol.obasis, "conventions")
        and hasattr(mol.obasis, "primitive_normalization")
        and hasattr(mol.obasis, "shells")
    ):
        raise ValueError("`mol` must be an IOData instance.")

    molbasis = mol.obasis

    cart_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "c"}
    sph_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "p"}

    # NOTE: hard-coded angular momentum from iodata.basis.ANGMOM_CHARS
    iodata_angmom = 'spdfghiklmnoqrtuvwxyzabce'

    class IODataShell(GeneralizedContractionShell):
        """Shell object that is compatible with gbasis' shell object.

        See `gbasis.contractions.GeneralizedContractionShell` for the documentation.

        """

        @property
        def angmom_components_cart(self):
            r"""Return the angular momentum components as ordered within the MolecularBasis.

            Returns
            -------
            angmom_components_cart : np.ndarray(L, 3)
                The x, y, and z components of the angular momentum vectors
                (:math:`\vec{a} = (a_x, a_y, a_z)` where :math:`a_x + a_y + a_z = \ell`).
                :math:`L` is the number of Cartesian contracted Gaussian functions for the given
                angular momentum, i.e. :math:`(angmom + 1) * (angmom + 2) / 2`

            """
            if self.angmom == 0:
                return np.array([[0, 0, 0]])
            return np.array(
                [(j.count("x"), j.count("y"), j.count("z")) for j in cart_conventions[self.angmom]]
            )

        @property
        def angmom_components_sph(self):
            """Return the ordering of the magnetic quantum numbers for the given angmom.

            Returns
            -------
            angmom_components_sph : tuple of int
                Tuple of magnetic quantum numbers of the contractions that specifies the
                ordering after transforming the contractions from the Cartesian to spherical
                coordinate system.

            """
            if self.angmom == 0:
                return (0,)
            raise NotImplementedError(
                "Iodata seems to be using 'Wikipedia-ordered real solid spherical harmonics', "
                "which isn't really documented anywhere. Until it's documented, spherical "
                "contractions will not be supported. Reference: {}".format(sph_conventions)
            )

    if molbasis.primitive_normalization != "L2":
        raise ValueError(
            "Only L2 normalization scheme is supported in gbasis. Given IOData instance uses "
            "primitive normalization scheme, {}".format(molbasis.primitive_normalization)
        )

    basis = []
    coord_types = []
    for grouped_shell in molbasis.shells:
        # segment the shell if not every contraction within the shell has the same angular
        # momentum and "kind"
        if len(set(grouped_shell.angmoms)) != 1 or len(set(grouped_shell.kinds)) != 1:
            shells = []
            for angmom, kind, coeffs in zip(
                grouped_shell.angmoms, grouped_shell.kinds, grouped_shell.coeffs.T
            ):
                shells.append(
                    # NOTE: _replace returns a copy (so the original is not affected)
                    grouped_shell._replace(
                        icenter=grouped_shell.icenter,
                        angmoms=[angmom],
                        kinds=[kind],
                        exponents=grouped_shell.exponents,
                        coeffs=coeffs.reshape(-1, 1),
                    )
                )
        else:
            shells = [grouped_shell]

        for shell in shells:
            # get angular momentum
            # NOTE: GeneralizedContractionShell only accepts angular momentum as an int.
            angmom = int(shell.angmoms[0])

            # get type
            coord_types.append(shell.kinds[0])

            # pylint: disable=E1136
            basis.append(
                IODataShell(angmom, mol.atcoords[shell.icenter], shell.coeffs, shell.exponents)
            )

    return basis
