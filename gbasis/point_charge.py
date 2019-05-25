"""Module for computing point charge integrals."""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
from scipy.special import hyp1f1  # pylint: disable=E0611


class PointChargeIntegral(BaseTwoIndexSymmetric):
    r"""General class for calculating one-electron integrals for interaction with a point charge.

    One electron integrals are assumed to have the following form:
    .. math::

        \int \phi_a(\mathbf{r}) \phi_b(\mathbf{r}) g(|\mathbf{r} - \mathbf{R}_C|, Z) d\mathbf{r}

    where :math:`\phi_a` is Gaussian contraction on the left, :math:`\phi_b` is the Gaussian
    contraction on the right, :math:`\mathbf{R}_C` is the position of the point charge, :math:`Z` is
    the charge at the point charge, and :math:`g(|\mathbf{r} - \mathbf{R}_C|, Z)` determines the
    type of the one-electorn integral.


    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first and second indices of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    boys_func : np.ndarray(np.ndarray(M, K_b, K_a))
        Boys function used to evaluate the one-electron integral.
        `M` is the number of orders that will be evaluated. `K_a` and `K_b` are the number of
        primitives on the left and right side, respectively.
    construct_array_contraction(self, contractions_one, contractions_two, coords_points,
                                charges_points)
        Return the point charge integrals for the given `ContractedCartesianGaussians` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the one-electron integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the one-electron integrals associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types, **kwargs) : np.ndarray(K_cont, K_cont)
        Return the one-electron integrals associated with the contraction in the given coordinate
        system.
        `K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the one-electron integrals associated with the linear combinations of contractions in
        the given coordinate system.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def boys_func(orders, weighted_dist):
        r"""Return the value of Boys function for the given orders and weighted distances.

        The Coulombic Boys function can be written as a renormalized special case of the Kummer
        confluent hypergeometric function, as derived in Helgaker (eq. 9.8.39).

        Parameters
        ----------
        orders : np.ndarray(M, 1, 1, 1)
            Differentiation order of the helper function.
            Same as m in eq. 23, Aldrichs, R. Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
            `M` is the number of orders that will be evaluated.
        weighted_dist : np.ndarray(1, N, K_b, K_a)
            Weighted interatomic distances.
            .. math::

                \frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{PC}||^2

            where :math:`\alpha_i` is the exponent of the ith primitive on the left side and the
            :math:`\beta_j` is the exponent of the jth primitive on the right side.
            `N` is the number of point charges. `K_a` and `K_b` are the number of primitives on the
            left and right side, respectively. Note that the index 2 corresponds to the primitive on
            the right side and the index 3 corresponds to the primitive on the left side.

        Returns
        -------
        boys_eval : np.ndarray(M, N, K_b, K_a)
            Output is the Boys function evaluated for each order and the weighted interactomic
            distance.

        Notes
        -----
        There's some documented instability for hyp1f1, mainly for large values or complex numbers.
        In this case it seems fine, since m should be less than 10 in most cases, and except for
        exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
        problem becomes a precision error in most cases where it was an overflow error before, so
        the values should be close even when they are wrong.

        To use another `boys_func`, simply overwrite this function (via monkeypatching or
        inheritance) with the desired boys function. Make sure to follow the same API, i.e. *have
        the same inputs including their shapes and types*. Note that the index `1` corresponds to
        the primitive on the right side and the index `2` correspond to the primitive on the left
        side.

        """
        return hyp1f1(orders + 1 / 2, orders + 3 / 2, -weighted_dist) / (2 * orders + 1)

    @classmethod
    def construct_array_contraction(
        cls, contractions_one, contractions_two, coords_points, charges_points
    ):
        """Return point charge interaction integral for the given contractions and point charges.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        coords_points : np.ndarray(N, 3)
            Coordinates of the point charges.
            Rows correspond to the different point charges and columns correspond to the x, y, and z
            components.
        charges_points : np.ndarray(N)
            Charge of the point charges.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, N)
            Point charge integral associated with the given instances of
            ContractedCartesianGaussians.
            First axis corresponds to the segmented contraction within `contractions_one`. `M_1` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the first index.
            Second axis corresponds to the angular momentum vector of the `contractions_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Third axis corresponds to the segmented contraction within `contractions_two`. `M_2` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the second index.
            Fourth axis corresponds to the angular momentum vector of the `contractions_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.
            Fifth axis cooresponds to the point charge. `N` is the number of points charges.

        Raises
        ------
        TypeError
            If `contractions_one` is not a ContractedCartesianGaussians instance.
            If `contractions_two` is not a ContractedCartesianGaussians instance.
            If `coords_points` is not a two-dimensional numpy array of dtype int/float with 3
            columns.
            If `charges_points` is not a one-dimensional numpy array of int/float.
        ValueError
            If `coords_points` does not have the same number of rows as the size of
            `charges_points`.

        """
        # pylint: disable=R0914

        if not isinstance(contractions_one, ContractedCartesianGaussians):
            raise TypeError("`contractions_one` must be a ContractedCartesianGaussians instance.")
        if not isinstance(contractions_two, ContractedCartesianGaussians):
            raise TypeError("`contractions_two` must be a ContractedCartesianGaussians instance.")
        if not (
            isinstance(coords_points, np.ndarray)
            and coords_points.ndim == 2
            and coords_points.shape[1] == 3
            and coords_points.dtype in [int, float]
        ):
            raise TypeError(
                "`coords_points` must be a two-dimensional numpy array of dtype int/float with "
                "three columns."
            )
        if not (
            isinstance(charges_points, np.ndarray)
            and charges_points.ndim == 1
            and charges_points.dtype in [int, float]
        ):
            raise TypeError(
                "`charges_points` must be a one-dimensional numpy array of integers or floats."
            )
        if coords_points.shape[0] != charges_points.size:
            raise ValueError(
                "`coords_points` must have the same number of rows as there are elements in "
                "`charges_points`."
            )

        # TODO: Overlap screening

        coord_a = contractions_one.coord
        angmom_a = contractions_one.angmom
        angmoms_a = contractions_one.angmom_components
        exps_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        coord_b = contractions_two.coord
        angmom_b = contractions_two.angmom
        angmoms_b = contractions_two.angmom_components
        exps_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs

        # Enforce L_a >= L_b
        ab_swapped = False
        if angmom_a < angmom_b:
            coord_a, coord_b = coord_b, coord_a
            angmom_a, angmom_b = angmom_b, angmom_a
            angmoms_a, angmoms_b = angmoms_b, angmoms_a
            exps_a, exps_b = exps_b, exps_a
            coeffs_a, coeffs_b = coeffs_b, coeffs_a
            ab_swapped = True

        integrals = _compute_one_elec_integrals(
            coords_points,
            cls.boys_func,
            coord_a,
            angmom_a,
            exps_a,
            coeffs_a,
            coord_b,
            angmom_b,
            exps_b,
            coeffs_b,
        )
        integrals = np.transpose(integrals, (7, 0, 1, 2, 8, 3, 4, 5, 6))

        angmoms_a_x, angmoms_a_y, angmoms_a_z = angmoms_a.T
        angmoms_b_x, angmoms_b_y, angmoms_b_z = angmoms_b.T

        # Generate output array
        # Ordering for output array:
        # axis 0 : index for segmented contractions of contraction one
        # axis 1 : angular momentum vector of contraction one (in the same order as angmoms_a)
        # axis 2 : index for segmented contractions of contraction two
        # axis 3 : angular momentum vector of contraction two (in the same order as angmoms_b)
        # axis 4 : point charge
        output = (
            -charges_points
            * integrals[
                np.arange(coeffs_a.shape[1])[:, None, None, None, None],
                angmoms_a_x[None, :, None, None, None],
                angmoms_a_y[None, :, None, None, None],
                angmoms_a_z[None, :, None, None, None],
                np.arange(coeffs_b.shape[1])[None, None, :, None, None],
                angmoms_b_x[None, None, None, :, None],
                angmoms_b_y[None, None, None, :, None],
                angmoms_b_z[None, None, None, :, None],
                np.arange(coords_points.shape[0])[None, None, None, None, :],
            ]
        )

        if ab_swapped:
            return np.transpose(output, (2, 3, 0, 1, 4))

        return output


def point_charge_cartesian(basis, coords_points, charges_points):
    """Return the integrals of the interaction b/w Cartesian contractions given point charges.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords_points : np.ndarray(N, 3)
        Coordinates of the point charges.
        Rows correspond to the different point charges and columns correspond to the x, y, and z
        components.
    charges_points : np.ndarray(N)
        Charge of the point charges.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart, N)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return PointChargeIntegral(basis).construct_array_cartesian(
        coords_points=coords_points, charges_points=charges_points
    )


def point_charge_spherical(basis, coords_points, charges_points):
    """Return the point-charge interation integrals of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords_points : np.ndarray(3,)
        Coordinate of the point charge.
        Rows correspond to the different point charges and columns correspond to the x, y, and z
        components.
    charges_points : float
        Charge of the point charge.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph, N)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return PointChargeIntegral(basis).construct_array_spherical(
        coords_points=coords_points, charges_points=charges_points
    )


def point_charge_mix(basis, coords_points, charges_points, coord_types):
    """Return the point-charge interation integrals of basis set in the given coordinate systems.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords_points : np.ndarray(3,)
        Coordinate of the point charge.
        Rows correspond to the different point charges and columns correspond to the x, y, and z
        components.
    charges_points : float
        Charge of the point charge.
    coord_types : list/tuple of str
        Types of the coordinate system for each ContractedCartesianGaussians.
        Each entry must be one of "cartesian" or "spherical".

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Array whose first and second indices are associated with the linear combinations of the
        contractions.
        First and second indices of the array correspond to the linear combination of contractions.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """
    return PointChargeIntegral(basis).construct_array_mix(
        coord_types, coords_points=coords_points, charges_points=charges_points
    )


def point_charge_lincomb(basis, transform, coords_points, charges_points, coord_type="spherical"):
    """Return the point-charge interaction integrals of the linera combination of basis set.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.
    coords_points : np.ndarray(N, 3)
        Coordinates of the point charges.
        Rows correspond to the different point charges and columns correspond to the x, y, and z
        components.
    charges_points : np.ndarray(N)
        Charge of the point charges.
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each ContractedCartesianGaussians instance.
        Default value is "spherical".

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs, N)
        Array whose first and second indices are associated with the linear combinations of the
        contractions.
        First and second indices of the array correspond to the linear combination of contractions.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """
    return PointChargeIntegral(basis).construct_array_lincomb(
        transform, coord_type, coords_points=coords_points, charges_points=charges_points
    )
