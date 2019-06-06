"""Integrals over differential operator involving contracted Cartesian Gaussians."""
from gbasis.integrals._moment_int import (
    _cleanup_intermediate_integrals,
    _compute_multipole_moment_integrals_intermediate,
)
import numpy as np


# FIXME: returns nan when exponent is zero
def _compute_differential_operator_integrals_intermediate(
    order_diff_max, coord_a, angmom_a_max, exps_a, coord_b, angmom_b_max, exps_b
):
    r"""Return the intermediate integrals over differential operators of two contractions.

    # TODO: equation

    Parameters
    ----------
    order_diff_max : int
        Maximum order of differentiation at which the recursion will stop.
        From a set of orders of differentiations, it should be the maximum order of differentiation
        along all dimensions.
        Zeroth order differentiation (i.e. overlap) is not supported.
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmom_a_max : int
        Maximum angular momentum component value of the left side at which the recursion will stop.
        From a set of angular momentum vectors, it should be the maximum angular momentum.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_a : np.ndarray(L_a, K_a)
        Normalization constants for the primitives in each contraction on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmom_b_max : int
        Maximum angular momentum component value of the right side at which the recursion will stop.
        From a set of angular momentum vectors, it should be the maximum angular momentum.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives on the right side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_b : np.ndarray(L_b, K_b)
        Normalization constants for the primitives in each contraction on the right side.

    Returns
    -------
    integrals : np.ndarray(max_d + 1, max_b + 1, max_a + 1, 3, K_b, K_a)
        Intermediate integrals for each differentiation order, angular momentum component, and
        coordinate.
        Dimension 0 corresponds to the order of the differentiations along the corresponding
        coordinate. `max_d` is the maximum order of differentiation, i.e. `order_diff_max`.
        Dimension 1 corresponds to the angular momentum component of contraction b in the
        corresponding coordinate. `max_b` is the maximum angular momentum component for the right
        side, i.e. `angmom_b_max`.
        Dimension 2 corresponds to the angular momentum component of contraction a in the
        corresponding coordinate. `max_a` is the maximum angular momentum component for the left
        side, i.e. `angmom_a_max`.
        Dimension 3 corresponds to the coordinate dimension, i.e. :math:`x, y, \text{and} z`.
        Dimension 4 corresponds to the index for the primitive in contraction b.
        Dimension 5 corresponds to the index for the primitive in contraction a.

    Raises
    ------
    IndexError
        If the `orders_diff` is all zeros, i.e. :math:`(0, 0, 0)`.

    """
    # pylint: disable=R0914
    integrals = np.zeros(
        (
            order_diff_max + 1,
            angmom_b_max + 1,
            angmom_a_max + order_diff_max + 1,
            3,
            exps_b.size,
            exps_a.size,
        )
    )

    # adjust axis
    coord_a = coord_a[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    coord_b = coord_b[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    exps_a = exps_a[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    # NOTE: coeffs_a and coeffs_b are not flattened because tensordot will be used at the end where
    # the primitives are transformed to contractions

    # compute overlap
    integrals[0, :, :, :, :, :] = _compute_multipole_moment_integrals_intermediate(
        np.zeros(3),
        0,
        coord_a,
        angmom_a_max + order_diff_max,
        exps_a,
        coord_b,
        angmom_b_max,
        exps_b,
    )[0, :, :, :, :, :]

    # recurse over order of differentiation
    i_range = np.arange(angmom_a_max + order_diff_max + 1)[None, None, :, None, None, None]

    integrals[1, :, 0, :, :, :] = 2 * exps_a.squeeze(axis=(0, 2)) * integrals[0, :, 1, :, :, :]
    integrals[1, :, 1:-1, :, :, :] = (
        2 * exps_a.squeeze(axis=0) * integrals[0, :, 2:, :, :, :]
        - i_range[0, 0:1, 1:-1, :, :, :] * integrals[0, :, :-2, :, :, :]
    )
    for k in range(1, order_diff_max):
        integrals[k + 1, :, 0, :, :, :] = (
            2 * exps_a.squeeze(axis=(0, 2)) * integrals[k, :, 1, :, :, :]
        )
        integrals[k + 1, :, 1:-1, :, :, :] = (
            2 * exps_a.squeeze(axis=0) * integrals[k, :, 2:, :, :, :]
            - i_range[0, 0, 1:-1, :, :, :] * integrals[k, :, :-2, :, :, :]
        )

    return integrals[:, :, : angmom_a_max + 1]


# FIXME: returns nan when exponent is zero
def _compute_differential_operator_integrals(
    orders_diff,
    coord_a,
    angmoms_a,
    exps_a,
    coeffs_a,
    norm_a,
    coord_b,
    angmoms_b,
    exps_b,
    coeffs_b,
    norm_b,
):
    r"""Return the integrals over differential operators of two contractions.

    Parameters
    ----------
    orders_diff : np.ndarray(D, 3)
        Orders of differentiation along each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders of
        the differentiation.
        Zeroth order differentiation (i.e. overlap) is not supported.
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmoms_a : np.ndarray(L_a, 3)
        Angular momentum vectors :math:`(\ell_x, \ell_y, \ell_z)` for the contractions on the left
        side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_a : np.ndarray(L_a, K_a)
        Normalization constants for the primitives in each contraction on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmoms_b : np.ndarray(L_b, 3)
        Angular momentum vectors :math:`(\ell_x, \ell_y, \ell_z)` for the contractions on the right
        side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives on the right side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_b : np.ndarray(L_b, K_b)
        Normalization constants for the primitives in each contraction on the right side.

    Returns
    -------
    integrals : np.ndarray(D, M_a, L_a, M_b, L_b)
        Integrals over differentiation operators associated with the given orders of differentiation
        and angular momentum vectors (contractions).
        Dimension 0 corresponds to the order of the differentiations. `D` is the number of orders of
        differentiations given.
        Dimension 1 corresponds to segmented contractions within the given generalized contraction
        `a`, (same exponents and angular momentum but different coefficients). `M_a` is the number
        of segmented contractions.
        Dimension 2 corresponds to the angular momentum vector a segmented contraction of
        generalized contraction `a`. `L_a` is the number of angular momentum vectors.
        Dimension 3 corresponds to segmented contractions within the given generalized contraction
        `b`, (same exponents and angular momentum but different coefficients). `M_b` is the number
        of segmented contractions.
        Dimension 4 corresponds to the angular momentum vector a segmented contraction of
        generalized contraction `b`. `L_b` is the number of angular momentum vectors.

    Raises
    ------
    IndexError
        If the `orders_diff` is all zeros, i.e. :math:`(0, 0, 0)`.

    """
    integrals = _compute_differential_operator_integrals_intermediate(
        np.max(orders_diff), coord_a, np.max(angmoms_a), exps_a, coord_b, np.max(angmoms_b), exps_b
    )

    integrals = _cleanup_intermediate_integrals(
        integrals, orders_diff, angmoms_a, coeffs_a, norm_a, angmoms_b, coeffs_b, norm_b
    )
    return integrals
