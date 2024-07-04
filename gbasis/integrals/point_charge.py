"""Module for computing point charge integrals."""
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._one_elec_int import _compute_one_elec_integrals
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
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `PointChargeIntegral`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    boys_func : np.ndarray(np.ndarray(M, K_b, K_a))
        Boys function used to evaluate the one-electron integral.
        `M` is the number of orders that will be evaluated.
        `K_a` and `K_b` are the number of primitives on the left and right side, respectively.
    construct_array_contraction(self, contractions_one, contractions_two, points_coords, points_charge)
        Return the point charge integrals for the given `GeneralizedContractionShell` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
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
            `N` is the number of point charges.
            `K_a` and `K_b` are the number of primitives on the left and right side,
            respectively.
            Note that the index 2 corresponds to the primitive on the right side and the index 3
            corresponds to the primitive on the left side.

        Returns
        -------
        boys_eval : np.ndarray(M, N, K_b, K_a)
            Output is the Boys function evaluated for each order and the weighted interatomic
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
        the same inputs including their shapes and types*. Note that the index `2` corresponds to
        the primitive on the right side and the index `3` corresponds to the primitive on the left
        side.

        """
        return hyp1f1(orders + 1 / 2, orders + 3 / 2, -weighted_dist) / (2 * orders + 1)

    @classmethod
    def construct_array_contraction(
        cls, contractions_one, contractions_two, points_coords, points_charge
    ):
        r"""Return point charge interaction integral for the given contractions and point charges.

        Parameters
        ----------
        contractions_one : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        points_coords : np.ndarray(N, 3)
            Cartesian coordinates of each point charge.
            Rows correspond to the different point charges and columns correspond to the
            :math:`x, y, \text{and} z` components.
        points_charge : np.ndarray(N)
            Charge of each point charge.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, N)
            Point charge integral associated with the given instances of
            `GeneralizedContractionShell`.
            Dimension 0 corresponds to the segmented contraction within `cont_one`. `M_1` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the first index.
            Dimension 1 corresponds to the angular momentum vector of the `cont_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Dimension 2 corresponds to the segmented contraction within `cont_two`.
            `M_2` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the second index.
            Dimension 3 corresponds to the angular momentum vector of the `cont_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.
            Dimension 4 corresponds to the point charge. `N` is the number of point charges.

        Raises
        ------
        TypeError
            If `contractions_one` is not a `GeneralizedContractionShell` instance.
            If `contractions_two` is not a `GeneralizedContractionShell` instance.
            If `points_coords` is not a two-dimensional `numpy` array of `dtype` int/float with 3
            columns.
            If `points_charge` is not a one-dimensional `numpy` array of int/float.
        ValueError
            If `points_coords` does not have the same number of rows as the size of
            `points_charge`.

        """
        # pylint: disable=R0914

        if not isinstance(contractions_one, GeneralizedContractionShell):
            raise TypeError("`contractions_one` must be a `GeneralizedContractionShell` instance.")
        if not isinstance(contractions_two, GeneralizedContractionShell):
            raise TypeError("`contractions_two` must be a `GeneralizedContractionShell` instance.")
        if not (
            isinstance(points_coords, np.ndarray)
            and points_coords.ndim == 2
            and points_coords.shape[1] == 3
            and points_coords.dtype in [int, float]
        ):
            raise TypeError(
                "`points_coords` must be a two-dimensional `numpy` array of `dtype` int/float with "
                "three columns."
            )
        if not (
            isinstance(points_charge, np.ndarray)
            and points_charge.ndim == 1
            and points_charge.dtype in [int, float]
        ):
            raise TypeError(
                "`points_charge` must be a one-dimensional `numpy` array of integers or floats."
            )
        if points_coords.shape[0] != points_charge.size:
            raise ValueError(
                "`points_coords` must have the same number of rows as there are elements in "
                "`points_charge`."
            )

        # TODO: Overlap screening

        coord_a = contractions_one.coord
        angmom_a = contractions_one.angmom
        angmoms_a = contractions_one.angmom_components_cart
        exps_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        coord_b = contractions_two.coord
        angmom_b = contractions_two.angmom
        angmoms_b = contractions_two.angmom_components_cart
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
            points_coords,
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
            -points_charge
            * integrals[
                np.arange(coeffs_a.shape[1])[:, None, None, None, None],
                angmoms_a_x[None, :, None, None, None],
                angmoms_a_y[None, :, None, None, None],
                angmoms_a_z[None, :, None, None, None],
                np.arange(coeffs_b.shape[1])[None, None, :, None, None],
                angmoms_b_x[None, None, None, :, None],
                angmoms_b_y[None, None, None, :, None],
                angmoms_b_z[None, None, None, :, None],
                np.arange(points_coords.shape[0])[None, None, None, None, :],
            ]
        )

        if ab_swapped:
            return np.transpose(output, (2, 3, 0, 1, 4))

        return output


def point_charge_integral(basis, points_coords, points_charge, transform=None):
    r"""Return the point-charge interaction integrals of basis set in the given coordinate systems.

    .. math::

        V_{ab} = \int \phi_a(\mathbf{r}) \frac{1}{|\mathbf{r} - \mathbf{R}_C|} \phi_b(\mathbf{r}) d\mathbf{r}

    where :math:`\mathbf{R}_C` is the position of the point charge :math:`C` and :math:`V_{ab}` is
    the interaction integral between the pair of basis functions :math:`\phi_a` and :math:`\phi_b`, and
    the point charge.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points_coords : np.ndarray(N, 3)
        Coordinates of the point charges (in atomic units).
        Rows correspond to the different point charges and columns correspond to the
        :math:`x, y, \text{and} z` components.
    points_charge : np.ndarray(N)
        Charge at each given point.
    transform : np.ndarray(K, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    eval_array : np.ndarray(K, K, N)
        Evaluations of the basis functions at the given coordinates.
        If keyword argument `transform` is provided, then the transformed basis functions will be
        evaluted at the given points.
        `K` is the total number of basis functions within the given basis set.
        `N` is the number of point charges.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if transform is not None:
        return PointChargeIntegral(basis).construct_array_lincomb(
            transform, coord_type, points_coords=points_coords, points_charge=points_charge
        )
    if all(ct == "cartesian" for ct in coord_type):
        return PointChargeIntegral(basis).construct_array_cartesian(
            points_coords=points_coords, points_charge=points_charge
        )
    if all(ct == "spherical" for ct in coord_type):
        return PointChargeIntegral(basis).construct_array_spherical(
            points_coords=points_coords, points_charge=points_charge
        )
    return PointChargeIntegral(basis).construct_array_mix(
        coord_type, points_coords=points_coords, points_charge=points_charge
    )
