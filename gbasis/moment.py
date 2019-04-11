"""Module for computing the moments of a basis set."""
from gbasis._moment_int import _compute_multipole_moment_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np


class Moment(BaseTwoIndexSymmetric):
    """Class for obtaining the moment for a set of Gaussian contractions.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the moment.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first and second indices of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction) : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
        Return the moment associated with a `ContractedCartesianGaussians` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the moment associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the moment associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the moment associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(
        contractions_one, contractions_two, moment_coord, moment_orders
    ):
        """Return the evaluations of the given contractions at the given coordinates.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        moment_coord : np.ndarray(3,)
            Center of the moment.
        moment_orders : np.ndarray(D, 3)
            Orders of the moment for each dimension (x, y, z).
            Note that a two dimensional array must be given, even if there is only one set of orders
            of the moment.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, D)
            Moment associated with the given instances of ContractedCartesianGaussians.
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
            Fifth axis corresponds to the orders of the moments. `D` is the number of orders for
            which the moment is computed.

        Raises
        ------
        TypeError
            If contractions_one is not a ContractedCartesianGaussians instance.
            If contractions_two is not a ContractedCartesianGaussians instance.
            If moment_coord is not a one-dimensional numpy array with 3 elements.
            If moment_orders is not a two-dimensional numpy array with 3 columns and dtype int.

        Notes
        -----
        Even though it will be faster to access the different orders of moments if it was associated
        with the first axis, the API will be consistent with the rest of the BaseTwoSymmetric class
        if the first four indices correspond to the segmented contraction and the angular momentum
        vector. If many orders of moments are calculated (the exact number depends on the size of
        the basis set), then it may be faster to access the moments if the fifth axis is moved back
        to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))` to change the
        order.

        """
        # pylint: disable=R0914
        if not isinstance(contractions_one, ContractedCartesianGaussians):
            raise TypeError("`contractions_one` must be a ContractedCartesianGaussians instance.")
        if not isinstance(contractions_two, ContractedCartesianGaussians):
            raise TypeError("`contractions_two` must be a ContractedCartesianGaussians instance.")
        if not (
            isinstance(moment_coord, np.ndarray)
            and moment_coord.ndim == 1
            and moment_coord.size == 3
        ):
            raise TypeError("`moment_coord` must be a one-dimensional numpy array with 3 elements.")
        if not (
            isinstance(moment_orders, np.ndarray)
            and moment_orders.ndim == 2
            and moment_orders.shape[1] == 3
            and moment_orders.dtype == int
        ):
            raise TypeError(
                "`moment_orders` must be a two-dimensional numpy array with 3 columns and dtype int"
            )

        coord_a = contractions_one.coord
        angmoms_a = contractions_one.angmom_components
        alphas_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        norm_a_prim = contractions_one.norm_prim
        coord_b = contractions_two.coord
        angmoms_b = contractions_two.angmom_components
        alphas_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs
        norm_b_prim = contractions_two.norm_prim
        output = _compute_multipole_moment_integrals(
            moment_coord,
            moment_orders,
            coord_a,
            angmoms_a,
            alphas_a,
            coeffs_a,
            norm_a_prim,
            coord_b,
            angmoms_b,
            alphas_b,
            coeffs_b,
            norm_b_prim,
        )
        return np.transpose(output, (1, 2, 3, 4, 0))


def moment_basis_cartesian(basis, moment_coord, moment_orders):
    """Return the moment of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    moment_coord : np.ndarray(3,)
        Center of the moment.
    moment_orders : np.ndarray(D, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders
        of the moment.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart, D)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    Notes
    -----
    If enough orders of moments are calculated (the exact number depends on the size of the basis
    set), then it may be faster to access the moments for each order if the fifth axis is moved back
    to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))` to change the
    order.

    """
    return Moment(basis).construct_array_cartesian(
        moment_coord=moment_coord, moment_orders=moment_orders
    )


def moment_basis_spherical(basis, moment_coord, moment_orders):
    """Return the moment of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    moment_coord : np.ndarray(3,)
        Center of the moment.
    moment_orders : np.ndarray(D, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders
        of the moment.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph, D)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    Notes
    -----
    If enough orders of moments are calculated (the exact number depends on the size of the basis
    set), then it may be faster to access the moments for each order if the fifth axis is moved back
    to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))` to change the
    order.

    """
    return Moment(basis).construct_array_spherical(
        moment_coord=moment_coord, moment_orders=moment_orders
    )


def moment_basis_spherical_lincomb(basis, transform, moment_coord, moment_orders):
    """Return the moment of the linear combination of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.
    moment_coord : np.ndarray(3,)
        Center of the moment.
    moment_orders : np.ndarray(D, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders
        of the moment.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs, D)
        Array whose first and second indices are associated with the linear combinations of the
        contracted spherical Gaussians.
        First and second indices of the array correspond to the linear combination of contracted
        spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
        combinations.

    Notes
    -----
    If enough orders of moments are calculated (the exact number depends on the size of the basis
    set), then it may be faster to access the moments for each order if the fifth axis is moved back
    to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))` to change the
    order.

    """
    return Moment(basis).construct_array_spherical_lincomb(
        transform, moment_coord=moment_coord, moment_orders=moment_orders
    )
