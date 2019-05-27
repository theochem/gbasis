"""Integral over differential operator involving Contracted Cartesian Gaussians."""
from gbasis._moment_int import _compute_overlap_helper
import numpy as np


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
    """Return the integrals over differential operators of two contractions.

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
        Angular momentum vectors (lx, ly, lz) for the contractions on the left side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_a : np.ndarray(L_a, K_a)
        Normalization constants for the primitives in each contraction on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmoms_b : np.ndarray(L_b, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the right side.
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
        First index corresponds to the order of the differentiations. `D` is the number of orders of
        differentiations given.
        Second index corresponds to segmented contractions within the given generalized contraction
        `a`, (same exponents and angular momentum but different coefficients). `M_a` is the number
        of segmented contractions.
        Third index corresponds to the angular momentum vector a segmented contraction of
        generalized contraction `a`. `L_a` is the number of angular momentum vectors.
        Fourth index corresponds to segmented contractions within the given generalized contraction
        `b`, (same exponents and angular momentum but different coefficients). `M_b` is the number
        of segmented contractions.
        Fifth index corresponds to the angular momentum vector a segmented contraction of
        generalized contraction `b`. `L_b` is the number of angular momentum vectors.

    Raises
    ------
    IndexError
        If the `orders_diff` is all zeros, i.e. `(0, 0, 0)`.

    """
    # pylint: disable=R0914
    # NOTE: following convention will be used to organize the axis of the multidimensional arrays
    # axis 0 = index for order of differentiation in the corresponding dimension (size: L_d^{max})
    # axis 1 = index for angular momentum component of contraction b in the corresponding dimension
    # (size: L_b^{max})
    # axis 2 = index for angular momentum component of contraction a in the corresponding dimension
    # (size: L_a^{max})
    # axis 3 = index for dimension (x, y, z) of coordinate (size: 3)
    # axis 4 = index for primitive of contraction b (size: K_b)
    # axis 5 = index for primitive of contraction a (size: K_a)

    angmom_a_max = np.max(angmoms_a)
    angmom_b_max = np.max(angmoms_b)
    order_diff_max = np.max(orders_diff)

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

    # sum of the exponents
    exps_sum = exps_a + exps_b
    # coordinate of the weighted average center
    coord_wac = (exps_a * coord_a + exps_b * coord_b) / exps_sum
    # relative distance from weighted average center
    rel_coord_a = coord_wac - coord_a
    rel_coord_b = coord_wac - coord_b
    # harmonic mean
    harm_mean = exps_a * exps_b / exps_sum

    # compute overlap
    integrals = _compute_overlap_helper(
        integrals,
        exps_sum,
        harm_mean,
        coord_a,
        coord_b,
        rel_coord_a,
        rel_coord_b,
        angmom_a_max + order_diff_max,
        angmom_b_max,
    )

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

    # select the appropriate orders and angular momentums
    integrals = integrals[
        orders_diff[:, np.newaxis, np.newaxis, :],
        angmoms_b[np.newaxis, :, np.newaxis, :],
        angmoms_a[np.newaxis, np.newaxis, :, :],
        np.arange(3)[np.newaxis, np.newaxis, np.newaxis, :],
    ]

    # multiply the x, y, and z components together
    integrals = np.prod(integrals, axis=3)
    # NOTE: axis 3 for dimension (x, y, z) of the coordinate has been removed

    # transform the primitives
    norm_a = norm_a[np.newaxis, np.newaxis, :, np.newaxis, :]
    integrals = np.tensordot(integrals * norm_a, coeffs_a, (4, 0))
    # NOTE: axis for primitive of contraction a has been removed (axis 4), and axis for segmented
    # contractions has been added at the right (axis 4)
    norm_b = norm_b[np.newaxis, :, np.newaxis, :, np.newaxis]
    integrals = np.tensordot(integrals * norm_b, coeffs_b, (3, 0))
    # NOTE: axis for primitive of contraction b (axis 3) has been removed, axis for segmented
    # contractions changes to axis 3, and axis for segmented contractions has been added at the
    # right (axis 4)
    # NOTE: now the
    # axis 0 = index for order of differentiation (size: D)
    # axis 1 = index for angular momentum component of contraction b in the corresponding dimension
    # (size: L_b^{max})
    # axis 2 = index for angular momentum component of contraction a in the corresponding dimension
    # (size: L_a^{max})
    # axis 3 = index for segmented contractions of contraction a (size: M_a)
    # axis 4 = index for segmented contractions of contraction b (size: M_b)
    return np.transpose(integrals, (0, 3, 2, 4, 1))
