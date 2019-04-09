"""Data classes for contracted Gaussians."""
import numpy as np
from scipy.special import factorial2


class ContractedCartesianGaussians:
    r"""Data class for contracted Cartesian Gaussians of the same angular momentum.

    .. math::

        \phi_{\vec{a}, A} (\mathbf{r}) &=
        \sum_i d_i (x - X_A)^{a_x} (y - Y_A)^{a_y} (z - Z_A)^{a_z}
        \exp{-\alpha_i |\vec{r} - \vec{R}_A|^2}\\
        &= \sum_i d_i g_{i} (\vec{r} | \vec{a}, \vec{R}_A)

    where :math:`\vec{r} = (x, y, z)`, :math:`\vec{R}_A = (X_A, Y_A, Z_A)`,
    :math:`\vec{a} = (a_x, a_y, a_z)`, and :math:`g_i` is a Gaussian primitive.

    Since the integrals involving these contractions are computed using recursive relations that
    modify the :math:`\vec{a}`, we group the primitives that share the same properties (i.e.
    :math:`\vec{R}_A` and :math:`\alpha_i`) except for the :math:`\vec{a}` in the hopes of
    vectorizing and storing repeating elements.

    Attributes
    ----------
    angmom : int
        Angular momentum of the set of contractions.
        .. math::

            l = \sum_i \vec{a} = a_x + a_y + a_z

    coord : np.ndarray(3,)
        Coordinate of the center of the Gaussian primitives.
    charge : float
        Charge at the center of the Gaussian primitives.
    coeffs : np.ndarray(K, M)
        Contraction coefficients, :math:`\{d_i\}`, of the primitives.
        First axis corresponds to the primitive and the second axis corresponds to the different
        segmented contractions (same exponents and angular momentum but different coefficients).
    exps : np.ndarray(K,)
        Exponents of the primitives, :math:`\{\alpha_i\}`.

    """

    def __init__(self, angmom, coord, charge, coeffs, exps):
        r"""Initialize a ContractedCartesianGaussians instance.

        Attributes
        ----------
        angmom : int
            Angular momentum of the set of contractions.
            .. math::

                \sum_i \vec{a} = a_x + a_y + a_z

        coord : np.ndarray(3,)
            Coordinate of the center of the Gaussian primitives.
        charge : float
            Charge at the center of the Gaussian primitives.
        coeffs : {np.ndarray(K,), np.ndarray(K, M)}
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.
            If a two-dimensional array is given, the first axis corresponds to the primitive and the
            second axis corresponds to the different contractions that have the same exponents (and
            angular momentum) but different coefficients.
            If a one-dimensional array is given, a newaxis will be inserted in the second dimension.
        exps : np.ndarray(K,)
            Exponents of the primitives, :math:`\{\alpha_i\}`.

        """
        self.angmom = angmom
        self.coord = coord
        self.charge = charge
        self.coeffs = coeffs
        self.exps = exps

    @property
    def charge(self):
        """Charge at the center of the Gaussian primitives.

        Returns
        -------
        charge : float
            Point charge at the center of the Gaussian primitive.

        """
        return self._charge

    @charge.setter
    def charge(self, charge):
        """Set the charge at the center of the Gaussian primitives.

        Parameters
        ----------
        charge : {float, int}
            Point charge at the center of the Gaussian primitive.

        Raises
        ------
        TypeError
            If charge is not given as an integer or a float.

        """
        if isinstance(charge, int):
            charge = float(charge)
        if not isinstance(charge, float):
            raise TypeError("Charge must be given as an integer or a float.")
        self._charge = charge

    @property
    def coord(self):
        """Coordinate of the center of the Gaussian primitives.

        Returns
        -------
        coord : float
            Coordinate of the center of the Gaussian primitive.

        """
        return self._coord

    @coord.setter
    def coord(self, coord):
        """Set the coordinate of the center of the Gaussian primitives.

        Parameters
        ----------
        coord : {float, int}
            Coordinate of the center of the Gaussian primitive.

        Raises
        ------
        TypeError
            If coord is not a numpy array of dimension 3.
            If coord does not have data type of int or float.

        """
        if not (isinstance(coord, np.ndarray) and coord.size == 3):
            raise TypeError("Coordinate must be given as a numpy array of dimension 3.")
        if coord.dtype == int:
            coord = coord.astype(float)
        if coord.dtype != float:
            raise TypeError("The data type of the coordinate must be int or float.")

        self._coord = coord

    @property
    def angmom(self):
        r"""Angular momentum of the contractions.

        Returns
        -------
        angmom : int
            Angular momentum of the set of contractions.
            .. math::

                \sum_i \vec{a} = a_x + a_y + a_z

        """
        return self._angmom

    @angmom.setter
    def angmom(self, angmom):
        r"""Set the angular momentum of the contractions.

        Parameters
        ----------
        angmom : int
            Angular momentum of the set of contractions.
            .. math::

                \sum_i \vec{a} = a_x + a_y + a_z

        Raises
        ------
        ValueError
            If angular momentum is not given as an integer.
            If angular momentum is not given as a positive integer.

        """
        if not isinstance(angmom, int):
            raise TypeError("Angular momentum must be given as an integer")
        if angmom < 0:
            raise ValueError("Angular momentum must be a positive integer.")
        self._angmom = angmom

    @property
    def exps(self):
        r"""Exponents of the Gaussian primitives.

        Returns
        -------
        exps : np.ndarray(K,)
            Exponents of the primitives, :math:`\{\alpha_i\}`.

        """
        return self._exps

    @exps.setter
    def exps(self, exps):
        r"""Set the exponents of the Gaussian primitives.

        Parameters
        ----------
        exps : np.ndarray(K,)
            Exponents of the primitives, :math:`\{\alpha_i\}`.

        Raises
        ------
        TypeError
            If exps does not have data type of float.
        ValueError
            If exps and coeffs are not arrays of the same size.

        """
        if not (isinstance(exps, np.ndarray) and exps.dtype == float):
            raise TypeError("Exponents must be given as a numpy array of data type float.")
        if hasattr(self, "_coeffs") and self.coeffs.shape[0] != exps.size:
            raise ValueError(
                "Exponents array must have the same number of elements as the number of rows "
                "in the two-dimensional coefficient matrix (for the generalized contractions)."
            )

        self._exps = exps

    @property
    def coeffs(self):
        r"""Contraction coefficients of the Gaussian primitives.

        Returns
        -------
        coeffs : np.ndarray(K, M)
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.
            First axis corresponds to the primitive and the second axis corresponds to the different
            segmented contractions (same exponents and angular momentum but different coefficients).

        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        r"""Set the contraction coefficients of the Gaussian primitives.

        Parameters
        ----------
        coeffs : {np.ndarray(K,), np.ndarray(K, M)}
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.
            If a two-dimensional array is given, the first axis corresponds to the primitive and the
            second axis corresponds to the different contractions that have the same exponents (and
            angular momentum) but different coefficients.
            If a one-dimensional array is given, a newaxis will be inserted in the second dimension.

        Raises
        ------
        TypeError
            If coeffs is not a numpy array of data type of float.
        ValueError
            If exps and coeffs are not arrays of the same size.

        """
        if not (isinstance(coeffs, np.ndarray) and coeffs.dtype == float):
            raise TypeError("Contraction coefficients must be a numpy array of data type float.")
        if hasattr(self, "_exps"):
            if coeffs.ndim not in [1, 2]:
                raise ValueError(
                    "Coefficients array must be given as a one- or two-dimensional array."
                )
            if coeffs.ndim == 2 and coeffs.shape[0] != self.exps.shape[0]:
                raise ValueError(
                    "Coefficients array for generalized contractions must be given as a two-"
                    "dimensional array with the same number of rows as the size of the exponents "
                    "array."
                )
            if coeffs.ndim == 1 and coeffs.shape != self.exps.shape:
                raise ValueError(
                    "Coefficients array for segmented contractions must be given as a one-"
                    "dimensional array with the same size as the exponents array."
                )
        if coeffs.ndim == 1:
            self._coeffs = coeffs[:, np.newaxis]
        else:
            self._coeffs = coeffs

    @property
    def angmom_components(self):
        r"""Components of the angular momentum.

        Returns
        -------
        angmom_components : np.ndarray(L, 3)
            The x, y, and z components of the angular momentum (:math:`\vec{a} = (a_x, a_y, a_z)`
            where :math:`a_x + a_y + a_z = l`).
            :math:`L` is the number of Cartesian contracted Gaussian functions for the given angular
            momentum, i.e. :math:`(angmom + 1) * (angmom + 2) / 2`

        """
        return np.array(
            [
                (x, y, self.angmom - x - y)
                for x in range(self.angmom + 1)
                for y in range(self.angmom - x + 1)
            ]
        )

    @property
    def norm(self):
        r"""Compute the normalization constant for a Cartesian Gaussian primitive.

            .. math::

                N(\vec{a}, \alpha) = (2 * \alpha / \pi)^{3/4} *
                (4 * \alpha)^{(a_x + a_y + a_z)/2} /
                ((2 * a_x - 1)!! * (2 * a_y - 1)!! * (2 * a_z - 1)!!)^{1/2}

        Returns
        -------
        norm : np.ndarray(L, K)
            The normalization constant of each of the Cartesian Gaussian primitives of the Cartesian
            contraction at each exponent.
            :math:`L` is the number of contracted Cartesian Gaussian functions for the given angular
            momentum, i.e. :math:`(angmom + 1) * (angmom + 2) / 2`

        """
        exponents = self.exps[np.newaxis, :]
        angmom_components = self.angmom_components[:, :, np.newaxis]

        return (
            (2 * exponents / np.pi) ** (3 / 4)
            * ((4 * exponents) ** (self.angmom / 2))
            / np.sqrt(np.prod(factorial2(2 * angmom_components - 1), axis=1))
        )

    @property
    def num_contr(self):
        """Return the number of Cartesian contracted Gaussian functions of given angular momentum.

        Returns
        -------
        num_contr : int
            Number of contracted Cartesian Gaussian functions of angular momentum, :math:`angmom`.

        """
        return (self.angmom + 1) * (self.angmom + 2) // 2


def make_contractions(basis_dict, atoms, coords, charges=None):
    """Return the contractions that correspond to the given atoms for the given basis.

    Parameters
    ----------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Output of the parsers from gbasis.parsers.
    atoms : N-list/tuple of str
        Atoms at which the contractions are centered.
    coords : np.ndarray(N, 3)
        Coordinates of each atom.
    charges : np.ndarray(N,)
        Charges of each atom.
        Default is 0 for each atom (neutral).

    Returns
    -------
    basis : tuple of ContractedCartesianGaussians
        Contractions for each atom.
        Contractions are ordered in the same order as in the values of `basis_dict`.

    Raises
    ------
    TypeError
        If atoms is not a list or tuple of strings.
        If coords is not a two-dimensional numpy array with 3 columns.
        If charges is not a one-dimensional numpy array.
    ValueError
        If the length of atoms is not equal to the number of rows of coords.
        If the length of charges is not equal to the length of atoms.

    """
    if not (isinstance(atoms, (list, tuple)) and all(isinstance(i, str) for i in atoms)):
        raise TypeError("Atoms must be provided as a list or tuple.")
    if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
        raise TypeError(
            "Coordinates must be provided as a two-dimensional numpy array with three columns."
        )
    if len(atoms) != coords.shape[0]:
        raise ValueError("Number of atoms must be equal to the number of rows in the coordinates.")

    if charges is None:
        charges = np.zeros(len(atoms))
    elif not (isinstance(charges, np.ndarray) and charges.ndim == 1):
        raise TypeError("Charges must be given as a one-dimensional numpy array.")
    if charges.size != len(atoms):
        raise ValueError("Number of charges must be equal to the number of atoms.")

    basis = []
    for atom, coord, charge in zip(atoms, coords, charges):
        for angmom, exps, coeffs in basis_dict[atom]:
            basis.append(ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps))
    return tuple(basis)
