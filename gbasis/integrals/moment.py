"""Module for computing the moments of a basis set."""
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals
import numpy as np


class Moment(BaseTwoIndexSymmetric):
    """Class for obtaining the moment for a set of Gaussian contractions.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the moment.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `Moment`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction) : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
        Return the moment associated with a `GeneralizedContractionShell` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the moment integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the moment integrals associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types, **kwargs) : np.ndarray(K_cont, K_cont)
        Return the moment integrals associated with all of the contraction in the given coordinate
        system.
        `K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform, coord_type, **kwargs) : np.ndarray(K_orbs, K_orbs)
        Return the moment integrals associated with the linear combinations of contractions in the
        given coordinate system.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(
        contractions_one, contractions_two, moment_coord, moment_orders
    ):
        """Return the evaluations of the given contractions at the given coordinates.

        Parameters
        ----------
        contractions_one : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : GeneralizedContractionShell
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
            Moment associated with the given instances of `GeneralizedContractionShell`.
            Dimension 0 corresponds to the segmented contraction within `contractions_one`.
            `M_1` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the first index.
            Dimension 1 corresponds to the angular momentum vector of the `contractions_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Dimension 2 corresponds to the segmented contraction within `contractions_two`.
            `M_2` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the second index.
            Dimension 3 corresponds to the angular momentum vector of the `contractions_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.
            Dimension 4 corresponds to the orders of the moments. `D` is the number of orders for
            which the moment is computed.

        Raises
        ------
        TypeError
            If contractions_one is not a `GeneralizedContractionShell` instance.
            If contractions_two is not a `GeneralizedContractionShell` instance.
            If moment_coord is not a one-dimensional `numpy` array with 3 elements.
            If moment_orders is not a two-dimensional `numpy` array with 3 columns and `dtype` int.

        Notes
        -----
        Even though it will be faster to access the different orders of moments if it was associated
        with the first axis, the API will be consistent with the rest of the `BaseTwoSymmetric`
        class if the first four indices correspond to the segmented contraction and the angular
        momentum vector. If many orders of moments are calculated (the exact number depends on the
        size of the basis set), then it may be faster to access the moments if the fifth axis is
        moved back to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))`
        to change the order.

        """
        # pylint: disable=R0914
        if not isinstance(contractions_one, GeneralizedContractionShell):
            raise TypeError("`contractions_one` must be a `GeneralizedContractionShell` instance.")
        if not isinstance(contractions_two, GeneralizedContractionShell):
            raise TypeError("`contractions_two` must be a `GeneralizedContractionShell` instance.")
        if not (
            isinstance(moment_coord, np.ndarray)
            and moment_coord.ndim == 1
            and moment_coord.size == 3
        ):
            raise TypeError(
                "`moment_coord` must be a one-dimensional `numpy` array with 3 elements."
            )
        if not (
            isinstance(moment_orders, np.ndarray)
            and moment_orders.ndim == 2
            and moment_orders.shape[1] == 3
            and moment_orders.dtype == int
        ):
            raise TypeError(
                "`moment_orders` must be a two-dimensional `numpy` array with 3 columns and `dtype`"
                " int"
            )

        coord_a = contractions_one.coord
        angmoms_a = contractions_one.angmom_components_cart
        exps_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        norm_a_prim = contractions_one.norm_prim_cart
        coord_b = contractions_two.coord
        angmoms_b = contractions_two.angmom_components_cart
        exps_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs
        norm_b_prim = contractions_two.norm_prim_cart
        output = _compute_multipole_moment_integrals(
            moment_coord,
            moment_orders,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a_prim,
            coord_b,
            angmoms_b,
            exps_b,
            coeffs_b,
            norm_b_prim,
        )
        return np.transpose(output, (1, 2, 3, 4, 0))


def moment_integral(basis, moment_coord, moment_orders, transform=None):
    """Return moment integral of the given basis set.

    .. math::

        \int \phi_a (\mathbf{r}) (x - X_C)^{c_x} (y - Y_C)^{c_y} (z - Z_C)^{c_z} \phi_b (\mathbf{r}) d\mathbf{r}

    where :math:`X_C`, :math:`Y_C`, and :math:`Z_C` are the coordinates of the center of the moment, and
    :math:`c_x`, :math:`c_y`, and :math:`c_z` are the orders of the moment.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    moment_coord : np.ndarray(3,)
        Center of the moment.
    moment_orders : np.ndarray(D, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders
        of the moment.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs, D)
        Moment integral of the given basis set.
        Dimensions 0 and 1 of the array correspond to the basis functions. `K_orbs` is the
        number of basis functions in the basis set.
        Dimension 2 of the array corresponds to the moment.
        `D` is the number of different moments.

    Notes
    -----
    If enough orders of moments are calculated (the exact number depends on the size of the basis
    set), then it may be faster to access the moments for each order if the fifth axis is moved back
    to the front prior to the access. Use `np.transpose(array, (4, 0, 1, 2, 3))` to change the
    order.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if transform is not None:
        return Moment(basis).construct_array_lincomb(
            transform, coord_type, moment_coord=moment_coord, moment_orders=moment_orders
        )
    if all(ct == "cartesian" for ct in coord_type):
        return Moment(basis).construct_array_cartesian(
            moment_coord=moment_coord, moment_orders=moment_orders
        )
    if all(ct == "spherical" for ct in coord_type):
        return Moment(basis).construct_array_spherical(
            moment_coord=moment_coord, moment_orders=moment_orders
        )
    return Moment(basis).construct_array_mix(
        coord_type, moment_coord=moment_coord, moment_orders=moment_orders
    )
