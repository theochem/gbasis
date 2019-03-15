"""Data classes for contracted Gaussians."""
from math import pi

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
    coeffs : np.ndarray(K,)
        Contraction coefficients, :math:`\{d_i\}`, of the primitives.
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
        coeffs : np.ndarray(K,)
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.
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
        exps : np.ndarray(L,)
            Exponents of the primitives, :math:`\{\alpha_i\}`.

        """
        return self._exps

    @exps.setter
    def exps(self, exps):
        r"""Set the exponents of the Gaussian primitives.

        Parameters
        ----------
        exps : np.ndarray(L,)
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
        if hasattr(self, "_coeffs") and exps.shape != self.coeffs.shape:
            raise ValueError("Exponents array must have the same size as Coefficients array")

        self._exps = exps

    @property
    def coeffs(self):
        r"""Contraction coefficients of the Gaussian primitives.

        Returns
        -------
        coeffs : np.ndarray(L,)
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.

        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        r"""Set the contraction coefficients of the Gaussian primitives.

        Parameters
        ----------
        coeffs : np.ndarray(L,)
            Contraction coefficients, :math:`\{d_i\}`, of the primitives.

        Raises
        ------
        TypeError
            If coeffs is not a numpy array of data type of float.
        ValueError
            If exps and coeffs are not arrays of the same size.

        """
        if not (isinstance(coeffs, np.ndarray) and coeffs.dtype == float):
            raise TypeError("Contraction coefficients must be a numpy array of data type float.")
        if hasattr(self, "_exps") and coeffs.shape != self.exps.shape:
            raise ValueError("Coefficients array must have the same size as exponents array.")

        self._coeffs = coeffs

    @property
    def angmom_components(self):
        r"""Components of the angular momentum.

        Returns
        -------
        angmom_components : np.ndarray(L, 3)
            The x, y, and z components of the angular momentum (:math:`\vec{a} = (a_x, a_y, a_z)`
            where :math:`a_x + a_y + a_z = l`).

        """
        return np.array(
            [
                (x, y, self.angmom - x - y)
                for x in range(self.angmom + 1)
                for y in range(self.angmom - x + 1)
            ]
        )

def cartesian_gaussian_norm(components, exponent):
    r"""Compute the normalization constant for a Cartesian Gaussian primitive.

        .. math::

            N(\vec{a}, \alpha) = (2 * \alpha / \pi)^{3/4} *
            (4 * \alpha)^{(a_x + a_y + a_z)/2} /
            ((2 * a_x - 1)!! * (2 * a_y - 1)!! * (2 * a_z - 1)!!)^{1/2}

    Parameters
    ----------
    components : np.ndarray(3,)
        The Cartesian components for angular momentum of the Gaussian primitive.
        .. math::

            \vec{a} = a_x + a_y + a_z

    exponent: float
        The exponent of the Cartesian Gaussian primitive.

    Returns
    -------
    norm : float
        The normalization constant of the Cartesian Gaussian primitive.

    """
    return (
        ((2 * exponent / pi) ** (3 / 4))
        * ((4 * exponent) ** (np.sum(components) / 2))
        / (np.sqrt(np.prod(factorial2(2 * components - 1))))
    )
