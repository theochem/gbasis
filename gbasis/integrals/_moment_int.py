"""Multipole moment integrals involving Contracted Cartesian Gaussians."""
import numpy as np


# FIXME: returns nan when exponent is zero
def _compute_multipole_moment_integrals_intermediate(
    coord_moment, order_moment_max, coord_a, angmom_a_max, exps_a, coord_b, angmom_b_max, exps_b
):
    """Return the intermediate multipole moment integrals of two contractions.

    # TODO: equation

    Parameters
    ----------
    coord_moment : np.ndarray(3,)
        Center of the moment.
    order_moment_max : int
        Maximum order of the moment at which the recursion will stop.
        From a set of orders of moments, it should be the maximum order of the moments along all
        dimensions.
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmom_a_max : int
        Maximum angular momentum component value of the left side at which the recursion will stop.
        From a set of angular momentum vectors, it should be the maximum angular momentum.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmom_b_max : int
        Maximum angular momentum component value of the right side at which the recursion will stop.
        From a set of angular momentum vectors, it should be the maximum angular momentum.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.

    Returns
    -------
    integrals : np.ndarray(D, M_a, L_a, M_b, L_b)
        Multipole moment integrals associated with the given orders of moments and angular momentum
        vectors (contractions).
        First index corresponds to the order of the moment. `D` is the number of moments given.
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

    Returns
    -------
    integrals : np.ndarray(max_d + 1, max_b + 1, max_a + 1, 3, K_b, K_a)
        Intermediate integrals for each moment order, angular momentum component, and coordinate.
        Dimension 0 corresponds to the order of the moments along the corresponding coordinate.
        `max_d` is the maximum order of differentiation, i.e. `order_diff_max`.
        Dimension 1 corresponds to the angular momentum component of contraction b in the
        corresponding coordinate. `max_b` is the maximum angular momentum component for the right
        side, i.e. `angmom_b_max`.
        Dimension 2 corresponds to the angular momentum component of contraction a in the
        corresponding coordinate. `max_a` is the maximum angular momentum component for the left
        side, i.e. `angmom_a_max`.
        Dimension 3 corresponds to the coordinate dimension, i.e. x, y, and z.
        Dimension 4 corresponds to the index for the primitive in contraction b.
        Dimension 5 corresponds to the index for the primitive in contraction a.

    """
    # pylint: disable=R0914
    integrals = np.zeros(
        (order_moment_max + 1, angmom_b_max + 1, angmom_a_max + 1, 3, exps_b.size, exps_a.size)
    )

    # adjust axis
    coord_moment = coord_moment[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
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
    rel_coord_moment = coord_wac - coord_moment
    # harmonic mean
    harm_mean = exps_a * exps_b / exps_sum

    # start of recursion
    integrals[0, 0, 0, :, :, :] = np.sqrt(np.pi / exps_sum) * np.exp(
        -harm_mean * (coord_a - coord_b) ** 2
    )

    # recurse over angular momentum for a
    # NOTE: array is sliced to avoid an if statement for angmom_a_max > 0
    integrals[0, 0, 1:2, :, :, :] = rel_coord_a * integrals[0, 0, 0:1, :, :, :]
    for i in range(1, angmom_a_max):
        integrals[0, 0, i + 1, :, :, :] = rel_coord_a * integrals[0, 0, i, :, :, :] + (
            i * integrals[0, 0, i - 1, :, :, :] / (2 * exps_sum)
        )

    # recurse over angular momentum for b
    i_range = np.arange(angmom_a_max + 1)[None, None, :, None, None, None]
    # NOTE: array is sliced to avoid an if statement for angmom_b_max > 0
    integrals[0, 1:2, 0, :, :, :] = rel_coord_b * integrals[0, 0:1, 0, :, :, :]
    integrals[0, 1:2, 1:, :, :, :] = rel_coord_b * integrals[0, 0:1, 1:, :, :, :] + (
        i_range[0, 0:1, 1:, :, :, :] * integrals[0, 0:1, :-1, :, :, :] / (2 * exps_sum)
    )
    for j in range(1, angmom_b_max):
        integrals[0, j + 1, 0, :, :, :] = rel_coord_b * integrals[0, j, 0, :, :, :] + (
            j * integrals[0, j - 1, 0, :, :, :] / (2 * exps_sum)
        )
        integrals[0, j + 1, 1:, :, :, :] = rel_coord_b * integrals[0, j, 1:, :, :, :] + (
            i_range[0, 0, 1:, :, :, :] * integrals[0, j, :-1, :, :, :]
            + j * integrals[0, j - 1, 1:, :, :, :]
        ) / (2 * exps_sum)

    # recurse over order of moment
    i_range = np.arange(angmom_a_max + 1)[None, None, :, None, None, None]
    j_range = np.arange(angmom_b_max + 1)[None, :, None, None, None, None]
    # NOTE: array is sliced to avoid an if statement for angmom_b_max > 0
    integrals[1:2, 0, 0, :, :, :] = rel_coord_moment * integrals[0:1, 0, 0, :, :, :]
    integrals[1:2, 0, 1:, :, :, :] = rel_coord_moment * integrals[0:1, 0, 1:, :, :, :] + (
        i_range[0:1, 0, 1:, :, :, :] * integrals[0:1, 0, :-1, :, :, :] / (2 * exps_sum)
    )
    integrals[1:2, 1:, 0, :, :, :] = rel_coord_moment * integrals[0:1, 1:, 0, :, :, :] + (
        j_range[0:1, 1:, 0, :, :, :] * integrals[0:1, :-1, 0, :, :, :] / (2 * exps_sum)
    )
    integrals[1:2, 1:, 1:, :, :, :] = rel_coord_moment * integrals[0:1, 1:, 1:, :, :, :] + (
        i_range[0:1, 0:1, 1:, :, :, :] * integrals[0:1, 1:, :-1, :, :, :]
        + j_range[0:1, 1:, 0:1, :, :, :] * integrals[0:1, :-1, 1:, :, :, :]
    ) / (2 * exps_sum)
    for k in range(1, order_moment_max):
        integrals[k + 1, 0, 0, :, :, :] = rel_coord_moment * integrals[k, 0, 0, :, :, :] + (
            k * integrals[k - 1, 0, 0, :, :, :] / (2 * exps_sum)
        )
        integrals[k + 1, 0, 1:, :, :, :] = rel_coord_moment * integrals[k, 0, 1:, :, :, :] + (
            i_range[0, 0, 1:, :, :, :] * integrals[k, 0, :-1, :, :, :]
            + k * integrals[k - 1, 0, 1:, :, :, :]
        ) / (2 * exps_sum)
        integrals[k + 1, 1:, 0, :, :, :] = rel_coord_moment * integrals[k, 1:, 0, :, :, :] + (
            j_range[0, 1:, 0, :, :, :] * integrals[k, :-1, 0, :, :, :]
            + k * integrals[k - 1, 1:, 0, :, :, :]
        ) / (2 * exps_sum)
        integrals[k + 1, 1:, 1:, :, :, :] = rel_coord_moment * integrals[k, 1:, 1:, :, :, :] + (
            i_range[0, 0:, 1:, :, :, :] * integrals[k, 1:, :-1, :, :, :]
            + j_range[0, 1:, 0:, :, :, :] * integrals[k, :-1, 1:, :, :, :]
            + k * integrals[k - 1, 1:, 1:, :, :, :]
        ) / (2 * exps_sum)

    return integrals


def _cleanup_intermediate_integrals(
    intermediate_integrals, orders, angmoms_a, coeffs_a, norm_a, angmoms_b, coeffs_b, norm_b
):
    """Return the intermediate integrals after selecting the appropriate components and normalizing.

    Parameters
    ----------
    integrals : np.ndarray(max_d + 1, max_b + 1, max_a + 1, 3, K_b, K_a)
        Intermediate integrals for each order, angular momentum component, and coordinate.
        Dimension 0 corresponds to the order along the corresponding coordinate. `max_d` is the
        maximum order, i.e. `order_diff_max`.
        Dimension 1 corresponds to the angular momentum component of contraction b in the
        corresponding coordinate. `max_b` is the maximum angular momentum component for the right
        side, i.e. `angmom_b_max`.
        Dimension 2 corresponds to the angular momentum component of contraction a in the
        corresponding coordinate. `max_a` is the maximum angular momentum component for the left
        side, i.e. `angmom_a_max`.
        Dimension 3 corresponds to the coordinate dimension, i.e. x, y, and z.
        Dimension 4 corresponds to the index for the primitive in contraction b.
        Dimension 5 corresponds to the index for the primitive in contraction a.
    orders : np.ndarray(D, 3)
        Orders for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders.
    angmoms_a : np.ndarray(L_a, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the left side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_a : np.ndarray(L_a, K_a)
        Normalization constants for the primitives in each contraction on the left side.
    angmoms_b : np.ndarray(L_b, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the right side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
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
        Integrals after selecting the appropriate orders and angular momentum components and after
        normalization.
        Dimension 0 corresponds to the order. `D` is the number of different orders given.
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

    """
    # select the appropriate moments and angular momentums
    integrals = intermediate_integrals[
        orders[:, np.newaxis, np.newaxis, :],
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
    # axis 0 = index for order of moment in the corresponding dimension (size: L_c^{max})
    # axis 1 = index for angular momentum component of contraction b in the corresponding dimension
    # (size: L_b^{max})
    # axis 2 = index for angular momentum component of contraction a in the corresponding dimension
    # (size: L_a^{max})
    # axis 3 = index for segmented contractions of contraction a (size: M_a)
    # axis 4 = index for segmented contractions of contraction b (size: M_b)
    return np.transpose(integrals, (0, 3, 2, 4, 1))


# FIXME: returns nan when exponent is zero
def _compute_multipole_moment_integrals(
    coord_moment,
    orders_moment,
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
    """Return the multipole moment integrals of two contractions.

    Parameters
    ----------
    coord_moment : np.ndarray(3,)
        Center of the moment.
    orders_moment : np.ndarray(D, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that a two dimensional array must be given, even if there is only one set of orders of
        the moment.
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
        Multipole moment integrals associated with the given orders of moments and angular momentum
        vectors (contractions).
        Dimension 0 corresponds to the order of the moment. `D` is the number of moments given.
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

    """
    integrals = _compute_multipole_moment_integrals_intermediate(
        coord_moment,
        np.max(orders_moment),
        coord_a,
        np.max(angmoms_a),
        exps_a,
        coord_b,
        np.max(angmoms_b),
        exps_b,
    )
    integrals = _cleanup_intermediate_integrals(
        integrals, orders_moment, angmoms_a, coeffs_a, norm_a, angmoms_b, coeffs_b, norm_b
    )
    return integrals
