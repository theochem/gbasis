"""Data classes for contracted Gaussians."""
import numpy as np


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

        """
        if not (isinstance(exps, np.ndarray) and exps.dtype == float):
            raise TypeError('Exponents must be given as a numpy array of data type float.')

        self._exps = exps
