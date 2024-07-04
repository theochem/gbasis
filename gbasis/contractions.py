"""Data class for contractions of Gaussian-type primitives."""

from numbers import Integral
import numpy as np
from gbasis.utils import factorial2


class GeneralizedContractionShell:
    r"""Data class for generalized contractions.

    Generalized contractions are defined to be a set of contractions that have the same angular
    momentum, center, and the same set of exponents. To avoid confusion with the term
    "contraction", we add "shell" to indicate that this instance refers to a collection of
    contractions.

    A Cartesian **primitive**, :math:`g_i`, is

    .. math::

        g_i (\mathbf{r} | \mathbf{R}_A, \mathbf{a}) =
        N_g (\alpha, \mathbf{a})
        (x - X_A)^{a_x} (y - Y_A)^{a_y} (z - Z_A)^{a_z}
        \exp{-\alpha_i |\mathbf{r} - \mathbf{R}_A|^2}

    where :math:`\mathbf{a} = (a_x, a_y, az)` is the angular momentum components in Cartesian
    coordinates, :math:`\mathbf{R}_A` is the coordinate of the center :math:`A`, :math:`N_g` is the
    normalization constant of the primitive, and :math:`\alpha_i` is the exponent of the primitive.

    A **Cartesian contraction** is a linear combination of Cartesian primitives:

    .. math::

        \phi (\mathbf{r} | \mathbf{R}_A, \mathbf{a}, \mathbf{d}, \boldsymbol{\alpha}) =
        N_{\phi} (\mathbf{R}_A, \mathbf{a}, \mathbf{d}, \boldsymbol{\alpha}) \sum_i d_i g_i (\mathbf{r} | \mathbf{R}_A, \mathbf{a})

    where :math:`d_i` is the contraction coefficient of the primitive and :math:`N_{\phi}` is the
    normalization constant of the contraction.

    Note that the Cartesian contraction depends on the angular momentum components in the Cartesian
    coordinate, :math:`\mathbf{a}`. At a given angular momentum, :math:`\ell`, we have
    :math:`\frac{(\ell + 1) (\ell + 2)}{2}` different components. We can linearly transform these
    contractions to obtain the :math:`2 \ell + 1` **spherical contractions**. Since we can convert
    the contractions in one coordinate system to another, we use the term **contraction** to
    describe both cases, unless otherwise stated.

    In order to obtain the spherical contractions for a given angular momentum, we need all of the
    contractions at that angular momentum (each with a different combination of angular momentum
    components. We will denote the collection of these contractions as a **segmented contraction
    shell**. Note that contractions within segmented contraction shell all have the same angular
    momentum, center, and the same set of exponents and contraction coefficients.

    Some basis sets, e.g. ANO-RCC, group together multiple segmented contractions that only
    differ by the contraction coefficients and angular momentums. Here, we denote a collection of
    contractions with the same angular momentum, center, and the same set of exponents as a
    **generalized contraction shell**. Note that **we do not support generalized contractions with
    different angular momentum**.

    We can think of a generalized contraction shell as a union of segmented contraction shells with
    the same angular momentum, center, and the same set of exponents. Now, the contraction
    coefficients depend on the specific segmented contraction shell, :math:`j`, to which the
    contraction belongs:

    .. math::

        \phi_j (\mathbf{r} | \mathbf{R}_A, \mathbf{a}, \mathbf{d}, \boldsymbol{\alpha}) =
        N_{\phi} (\mathbf{R}_A, \mathbf{a}, \mathbf{d}, \boldsymbol{\alpha}) \sum_i d_{ij} g_i (\mathbf{r} | \mathbf{R}_A, \mathbf{a})

    Attributes
    ----------
    coord : np.ndarray(3,)
        Coordinate of the center of the contractions.
    angmom : int
        Angular momentum of the contractions.

        .. math::
            \ell = \sum_i (\mathbf{a})_i = a_x + a_y + a_z

    angmom_components_cart : np.ndarray(L, 3)
        The x, y, and z components of the angular momentum vectors
        (:math:`\mathbf{a} = (a_x, a_y, a_z)` where :math:`a_x + a_y + a_z = \ell`).
        `L` is the number of Cartesian contracted Gaussian functions for the given angular
        momentum, i.e. :math:`(\ell + 1) * (\ell + 2) / 2`.
        Property of `GeneralizedContractionShell`.
    angmom_components_sph : tuple of int
        Tuple of magnetic quantum numbers of the contractions that specifies the ordering after
        transforming the contractions from the Cartesian to spherical coordinate system.
        Property of `GeneralizedContractionShell`.
    exps : np.ndarray(K,)
        Exponents of the primitives, :math:`\{\alpha_i\}_{i=1}^K`.
    coeffs : np.ndarray(K, M)
        Contraction coefficients, :math:`\{d_{ij}\}`, of the primitives.
        First axis corresponds to the primitive and the second axis corresponds to the segmented
        contraction shell.
    coord_type : str
        Type of the coordinate system used to specify the contractions.
    norm_cont : np.ndarray(M, L)
        Normalization constants of the Cartesian contractions of different angular momentum
        components and segmented contraction shells.
    norm_prim_cart : np.ndarray(L, K)
        The normalization constants of the Cartesian Gaussian primitives.
        `L` is the number of contracted Cartesian Gaussian functions for the given angular
        momentum, i.e. :math:`(\ell + 1) * (\ell + 2) / 2`.
        Property of `GeneralizedContractionShell`.
    num_cart : int
        Number of Cartesian contractions of angular momentum, :math:`\ell`.
        Property of `GeneralizedContractionShell`.
    num_sph : int
        Number of spherical contractions of angular momentum, :math:`\ell`.
        Property of `GeneralizedContractionShell`.

    Methods
    -------
    assign_norm_contr(self)
        Assign normalization constants for the contractions.

    """

    def __init__(self, angmom, coord, coeffs, exps, coord_type, icenter=None):
        r"""Initialize a GeneralizedContractionShell instance.

        Parameters
        ----------
        angmom : int
            Angular momentum of the set of contractions.

            :math:`\sum_i \mathbf{a} = a_x + a_y + a_z`

        coord : np.ndarray(3,)
            Coordinate of the center of the contractions.
        coeffs : {np.ndarray(K,), np.ndarray(K, M)}
            Contraction coefficients, :math:`\{d_{ij}\}`, of the primitives.
            If a two-dimensional array is given, the first axis corresponds to the primitive and the
            second axis corresponds to the segmented contraction shell.
            If a one-dimensional array is given, a `newaxis` will be inserted in the second
            dimension.
        exps : np.ndarray(K,)
            Exponents of the primitives, :math:`\{\alpha_i\}_{i=1}^K`.
        coord_type : str
            Coordinate type of the contraction. Options include "cartesian" or "c" and
            "spherical" or "p".
        icenter : np.int64 or None (optional)
            Index for the atomic center for the contraction
        """
        self.angmom = angmom
        self.coord = coord
        self.coeffs = coeffs
        self.exps = exps
        self.assign_norm_cont()
        self.coord_type = coord_type
        self.icenter = icenter

    @property
    def icenter(self):
        """Atom center index for the contractions.

        Returns
         -------
         icenter : int or None
             Index for the corresponding atom center of the contractions.

        """
        return self._icenter

    @icenter.setter
    def icenter(self, icenter):
        """Atom center index for the contractions.

        Parameters
        ----------
        icenter : int or None
            Index for the corresponding atom center of the contractions.

        Raises
        ------
        TypeError
            If `center` is not an `Integral` or `None` type.

        """
        if isinstance(icenter, Integral):
            self._icenter = int(icenter)
        elif icenter is None:
            self._icenter = None
        else:
            raise TypeError(f"Center should be of `numbers.Integral` type. Got {type(icenter)}")

    @property
    def coord(self):
        """Coordinate of the center of the contractions.

        Returns
        -------
        coord : np.ndarray(3,)
            Coordinate of the center of the contractions.

        """
        return self._coord

    @coord.setter
    def coord(self, coord):
        """Set the coordinate of the center of the contractions.

        Parameters
        ----------
        coord : np.ndarray(3,)
            Coordinate of the center of the contractions.

        Raises
        ------
        TypeError
            If `coord` is not a `numpy` array of dimension 3.
            If `coord` does not have data type of int or float.

        """
        if not (isinstance(coord, np.ndarray) and coord.size == 3):
            raise TypeError("Coordinate must be given as a `numpy` array of dimension 3.")
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

            :math:`\sum_i (\mathbf{a})_i = a_x + a_y + a_z`

        """
        return self._angmom

    @angmom.setter
    def angmom(self, angmom):
        r"""Set the angular momentum of the contractions.

        Parameters
        ----------
        angmom : int
            Angular momentum of the set of contractions.

            :math:`\sum_i (\mathbf{a})_i = a_x + a_y + a_z`

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
            Exponents of the primitives, :math:`\{\alpha_i\}_{i=1}^K`.

        """
        return self._exps

    @exps.setter
    def exps(self, exps):
        r"""Set the exponents of the Gaussian primitives.

        Parameters
        ----------
        exps : np.ndarray(K,)
            Exponents of the primitives, :math:`\{\alpha_i\}_{i=1}^K`.

        Raises
        ------
        TypeError
            If `exps` does not have data type of float.
        ValueError
            If `exps` and `coeffs` are not arrays of the same size.

        """
        if not (isinstance(exps, np.ndarray) and exps.dtype == float):
            raise TypeError("Exponents must be given as a `numpy` array of data type float.")
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
            Contraction coefficients, :math:`\{d_{ij}\}`, of the primitives.
            Dimension 0 corresponds to the primitive and the dimension 1 corresponds to the
            segmented contraction shell.

        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        r"""Set the contraction coefficients of the Gaussian primitives.

        Parameters
        ----------
        coeffs : {np.ndarray(K,), np.ndarray(K, M)}
            Contraction coefficients, :math:`\{d_{ij}\}`, of the primitives.
            If a two-dimensional array is given, the dimension 0 corresponds to the primitive and
            dimension 1 corresponds to the segmented contraction shell.
            If a one-dimensional array is given, a `newaxis` will be inserted in the second
            dimension.

        Raises
        ------
        TypeError
            If `coeffs` is not a `numpy` array of data type of float.
        ValueError
            If `coeffs` is not a one or two dimensional array.
            If `coeffs` refers to a generalized contraction (i.e. two dimensional array) and does
            not have the same number of rows as there are exponents (i.e. primitives).
            If `coeffs` refers to segmented contraction (i.e. one dimensional array) and does not
            have the same number of elements as there are exponents (i.e. primitives).

        """
        if not (isinstance(coeffs, np.ndarray) and coeffs.dtype == float):
            raise TypeError("Contraction coefficients must be a `numpy` array of data type float.")
        if coeffs.ndim not in [1, 2]:
            raise ValueError("Coefficients array must be given as a one- or two-dimensional array.")
        if hasattr(self, "_exps"):
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
    def angmom_components_cart(self):
        r"""Return the components of the angular momentum vectors for the given angular momentum.

        Returns
        -------
        angmom_components_cart : np.ndarray(L, 3)
            The x, y, and z components of the angular momentum vectors
            (:math:`\mathbf{a} = (a_x, a_y, a_z)` where :math:`a_x + a_y + a_z = \ell`).
            `L` is the number of Cartesian contracted Gaussian functions for the given
            angular momentum, i.e. :math:`(\ell + 1) * (\ell + 2) / 2`

        Notes
        -----
        In `HORTON2`, :math:`(a_x, a_y, a_z)` is represented by a string of X, Y, and Z characters,
        such that :math:`a_x` `X` characters, :math:`a_y` `Y` characters, and :math:`a_z` `Z`
        characters appear. For example, `(2, 1, 1)` corresponds to `XXYZ`

        """
        return np.array(
            [
                (x, y, self.angmom - x - y)
                for x in range(self.angmom, -1, -1)
                for y in range(self.angmom - x, -1, -1)
            ]
        )

    @property
    def angmom_components_sph(self):
        r"""
        Return the ordering of the magnetic quantum numbers for the given angular momentum.

        We define spherical components as regular solid harmonics :math:`R_{\ell, m}` such that

        .. math::

                R_{\ell, m} = C_{\ell, m}, \quad m = 0 ... \ell\\
                R_{\ell, -m} = S_{\ell, m}, \quad m = 1 ... \ell

        where :math:`C_{\ell, m}` and :math:`S_{\ell, m}` are Cosine- and Sine-like real
        regular solid harmonics (respectively).

        We will use these definitions to represent pure (spherical) functions as strings of the form
        `c{m}` and `s{m}`. By this definition, for pure f functions, the ordering by magnetic
        quantum number [0, 1, -1, 2, -2, 3, -3] corresponds to
        ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3'].

        Some quantum chemistry codes use different conventions that cause compatibility issues. For
        these conventions, it may be necessary to change the sign of specific pure functions,
        denoted by a minus sign, e.g. -c3. A primary example is for Molden files written by ORCA;
        these files use the conventions ['c0', 'c1', 's1', 'c2', 's2', '-c3', '-s3'].


        Returns
        -------
        angmom_components_sph : tuple of str
            Tuple of strings representing the cosine- and sine-like real regular solid harmonics
            representing the magnetic quantum numbers of the contractions specifying the ordering
            after transforming the contractions from the Cartesian to spherical coordinate system.

        Notes
        -----
        Our default ordering is from -m to m, e.g. for :math:`\ell = 2`,
        ['s2', 's1', 'c0', 'c1', 'c2'].

        """
        if self.angmom == 1:
            return tuple(["c1", "s1", "c0"])
        else:
            return tuple(
                ["s{}".format(m) for m in range(self.angmom, 0, -1)]
                + ["c{}".format(m) for m in range(self.angmom + 1)]
            )

    @property
    def norm_prim_cart(self):
        r"""Return the normalization constants of the Cartesian Gaussian primitives.

        For a Cartesian primitive with exponent :math:`\alpha_i`, the normalization constant is:

        .. math::
           N(\alpha_i, \mathbf{a}) = \sqrt {
           \left(\frac{2\alpha_i}{\pi}\right)^\frac{3}{2}
           \frac{(4\alpha_i)^{a_x + a_y + a_z}}{(2a_x - 1)!! (2a_y - 1)!! (2a_z - 1)!!}}

        Returns
        -------
        norm_prim_cart : np.ndarray(L, K)
            The normalization constants of the Cartesian Gaussian primitives.
            `L` is the number of contracted Cartesian Gaussian functions for the given angular
            momentum, i.e. :math:`(\ell + 1) * (\ell + 2) / 2`
            `K` is the number of exponents (i.e. primitives).

        """
        exponents = self.exps[np.newaxis, :]
        angmom_components_cart = self.angmom_components_cart[:, :, np.newaxis]

        return (
            (2 * exponents / np.pi) ** (3 / 4)
            * ((4 * exponents) ** (self.angmom / 2))
            / np.sqrt(np.prod(factorial2(2 * angmom_components_cart - 1), axis=1))
        )

    @property
    def num_cart(self):
        r"""Return the number of Cartesian contractions of the given angular momentum.

        .. math:: \frac{(\ell + 1)(\ell + 2)}{2}

        Returns
        -------
        num_cart : int
            Number of Cartesian contractions of angular momentum, `angmom`.

        """
        return (self.angmom + 1) * (self.angmom + 2) // 2

    @property
    def num_sph(self):
        r"""Return the number of spherical contractions of the given angular momentum.

        .. math:: 1 + 2\ell

        Returns
        -------
        num_sph : int
            Number of spherical contractions of angular momentum, `angmom`.

        """
        return 1 + 2 * self.angmom

    @property
    def num_seg_cont(self):
        """Return the number of segmented contractions.

        Returns
        -------
        num_seg_cont : int
            Number of segmented contractions.

        """
        return self.coeffs.shape[1]

    def assign_norm_cont(self):
        r"""Store the normalization constants of the contractions.

        .. math::
            \int \phi_i(\mathbf{r}) \phi_i(\mathbf{r}) d\mathbf{r} &= N\\
            \frac{1}{N} \int \phi_i(\mathbf{r}) \phi_i(\mathbf{r}) d\mathbf{r} &= 1\\
            \int \left(\frac{1}{\sqrt{N}} \phi_i(\mathbf{r})\right)
            \left(\frac{1}{\sqrt{N}} \phi_i(\mathbf{r})\right) d\mathbf{r} &= 1\\

        where :math:`N` is the norm of contraction :math:`\phi_i`.

        Notes
        -----
        Since the `overlap` module also depends on this (`contractions`) module, if the `overlap`
        module is imported at the start of this module, there is a circular import error. Therefore,
        the overlap module must be imported within this method.

        """
        from gbasis.integrals.overlap import Overlap  # pylint: disable=R0401,C0415

        self.norm_cont = np.einsum("ijij->ij", Overlap.construct_array_contraction(self, self))
        self.norm_cont **= -0.5

    @property
    def coord_type(self):
        """Return the coordinate type.

        Returns
        -------
        coord_type : str
            Coordinate type of the contraction.

        """
        return self._coord_type

    @coord_type.setter
    def coord_type(self, coord_type):
        """Set the coordinate type.

        Parameters
        ----------
        coord_type : str
            Coordinate type of the contraction.

        Raises
        ------
        TypeError
            If `coord_type` is not a string.
        ValueError
            If `coord_type` is not one of the following:
                - 'c' (cartesian)
                - 'cartesian'
                - 'p' (spherical)
                - 'spherical'
        """

        if not isinstance(coord_type, str):
            raise TypeError("Coordinate type must be given as a string.")
        if coord_type not in ["c", "cartesian", "p", "spherical"]:
            raise ValueError(
                "`coord_type` is incorrectly specified. It must be either 'c' "
                "or 'cartesian' for Cartesian coordinates, or 'p' or 'spherical' "
                "for spherical coordinates."
            )
        self._coord_type = {
            "c": "cartesian",
            "cartesian": "cartesian",
            "spherical": "spherical",
            "p": "spherical",
        }[coord_type]
