"""Module for interfacing to other quantum chemistry packages."""
from gbasis.contractions import GeneralizedContractionShell
import numpy as np


def from_iodata(mol):
    """Return basis set stored within the `IOData` instance in `iodata`.

    Parameters
    ----------
    mol : iodata.iodata.IOData
        `IOData` instance from `iodata` module.

    Returns
    -------
    basis : tuple of gbasis.contraciton.GeneralizedContractionShell
        Basis set object used within the `gbasis` module.
        `GeneralizedContractionShell` corresponds to the `Shell` object within `iodata.basis`.

    Raises
    ------
    NotImplementedError
        If any contractions in the given `IOData` instance is spherical.
    ValueError
        If `mol` is not an `iodata.iodata.IOData` instance.
        If the primitive normalization scheme of the shells in `IOData` instance is not "L2".

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

    molbasis = mol.obasis

    cart_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "c"}
    sph_conventions = {i[0]: j for i, j in molbasis.conventions.items() if i[1] == "p"}

    # NOTE: hard-coded angular momentum from iodata.basis.ANGMOM_CHARS
    iodata_angmom = "spdfghiklmnoqrtuvwxyzabce"

    class IODataShell(GeneralizedContractionShell):
        """Shell object that is compatible with `gbasis`' shell object.

        See `gbasis.contractions.GeneralizedContractionShell` for the documentation.

        """

        @property
        def angmom_components_cart(self):
            r"""Return the angular momentum components as ordered within the `MolecularBasis`.

            Returns
            -------
            angmom_components_cart : np.ndarray(L, 3)
                The x, y, and z components of the angular momentum vectors
                (:math:`\vec{a} = (a_x, a_y, a_z)` where :math:`a_x + a_y + a_z = \ell`).
                `L` is the number of Cartesian contracted Gaussian functions for the given
                angular momentum, i.e. :math:`(angmom + 1) * (angmom + 2) / 2`

            """
            if self.angmom == 0:
                return np.array([[0, 0, 0]])
            return np.array(
                [(j.count("x"), j.count("y"), j.count("z")) for j in cart_conventions[self.angmom]]
            )

        @property
        def angmom_components_sph(self):
            """Return the ordering of the magnetic quantum numbers for the given angular momentum.

            Returns
            -------
            angmom_components_sph : tuple of int
                Tuple of magnetic quantum numbers of the contractions that specifies the
                ordering after transforming the contractions from the Cartesian to spherical
                coordinate system.

            Raises
            ------
            NotImplementedError
                If convention requires multiplication by a negative sign for some of the harmonics.
                If convention does not have angular momentum character that matches up with the
                angular momentum.
            ValueError
                If convention does not support given angular momentum.
                If convention for the sign of the magnetic quantum number is not cosine (+1) or sine
                (-1).


            """
            output = []
            if self.angmom not in sph_conventions:
                raise ValueError(
                    "Given convention does not support spherical contractions for the angular "
                    "momentum {0}".format(self.angmom)
                )

            for j in sph_conventions[self.angmom]:
                if j[0] == "-":  # pragma: no cover
                    raise NotImplementedError(
                        "Only the real solid harmonics as defined in Helgaker Section 6.4.2. is "
                        "supported."
                    )
                if j[0] != iodata_angmom[self.angmom]:  # pragma: no cover
                    raise NotImplementedError(
                        "The angular momentum character, {0}, does not match up with the given "
                        "angular momentum, {1}".format(j[0], self.angmom)
                    )

                if j[1] == "c":
                    factor = 1
                elif j[1] == "s":
                    factor = -1
                else:  # pragma: no cover
                    raise ValueError(
                        "Convention for the sign of the magnetic quantum number must be either 'c' "
                        "or 's'."
                    )

                output.append(factor * int(j[2:]))

            return tuple(output)

    if molbasis.primitive_normalization != "L2":  # pragma: no cover
        raise ValueError(
            "Only L2 normalization scheme is supported in `gbasis`. Given `IOData` instance uses "
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

    basis = []
    for atom, coord in mol._atom:
        basis_info = mol._basis[atom]

        for shell in basis_info:
            angmom = shell[0]

            exps_coeffs = np.vstack(shell[1:])
            exps = exps_coeffs[:, 0]

            coeffs = exps_coeffs[:, 1:]

            basis.append(
                GeneralizedContractionShell(
                    angmom, np.array(coord), np.array(coeffs), np.array(exps)
                )
            )

    return tuple(basis)
