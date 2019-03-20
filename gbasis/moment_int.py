"""Multipole moment integrals involving Contracted Cartesian Gaussians."""
import numpy as np


# TODO: in the case of generalized Cartesian contraction where multiple shells have the same sets of
# exponents but different sets of primitive coefficients, it will be helpful to vectorize the
# `prim_coeffs` also.
def _compute_multipole_moment_integrals(
    coord_moment,
    orders_moment,
    coord_a,
    angmoms_a,
    alphas_a,
    coeffs_a,
    coord_b,
    angmoms_b,
    alphas_b,
    coeffs_b,
):
    """Return the multipole moment integrals of two contraction.

    Parameters
    ----------
    coord_moment : np.ndarray(3,)
        Center of the moment.
    orders_moment : np.ndarray(M, 3)
        Orders of the moment for each dimension (x, y, z).
        Note that two dimensional array must be given, even if there is only one set of orders of
        the moment.
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmoms_a : np.ndarray(L_a, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the left side.
        Note that two dimensional array must be given, even if there is only one angular momentum
        vector.
    alphas_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a,)
        Contraction coefficients of the primitives on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmoms_b : np.ndarray(L_b, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the right side.
        Note that two dimensional array must be given, even if there is only one angular momentum
        vector.
    alphas_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.
    coeffs_b : np.ndarray(K_b,)
        Contraction coefficients of the primitives on the right side.

    Returns
    -------
    integrals : np.ndarray(M, L_a, L_b)
        Multipole moment integrals associated with the given orders of moments and angular momentum
        vectors (contractions).

    """
    # pylint: disable=R0914
    # NOTE: following convention will be used to organize the axis of the multidimensional arrays
    # axis 0 = index for order of moment in the corresponding dimension (size: L_c^{max})
    # axis 1 = index for angular momentum component of contraction b in the corresponding dimension
    # (size: L_b^{max})
    # axis 2 = index for angular momentum component of contraction a in the corresponding dimension
    # (size: L_a^{max})
    # axis 3 = index for dimension (x, y, z) of coordinate (size: 3)
    # axis 4 = index for primitive of contraction a (size: K_a)
    # axis 5 = index for primitive of contraction b (size: K_b)

    angmom_a_max = np.max(angmoms_a)
    angmom_b_max = np.max(angmoms_b)
    order_moment_max = np.max(orders_moment)

    integrals = np.zeros(
        (order_moment_max + 1, angmom_b_max + 1, angmom_a_max + 1, 3, alphas_a.size, alphas_b.size)
    )

    # adjust axis
    coord_moment = coord_moment[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    coord_a = coord_a[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    coord_b = coord_b[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    alphas_a = alphas_a[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    alphas_b = alphas_b[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    # NOTE: coeffs_a and coeffs_b are not flattened because tensordot will be used at the end where
    # the primitives are transformed to contractions

    # sum of the exponents
    alphas_sum = alphas_a + alphas_b
    # coordinate of the weighted average center
    coord_wac = (alphas_a * coord_a + alphas_b * coord_b) / alphas_sum
    # relative distance from weighted average center
    rel_coord_a = coord_wac - coord_a
    rel_coord_b = coord_wac - coord_b
    rel_coord_moment = coord_wac - coord_moment
    # harmonic mean
    harm_mean = alphas_a * alphas_b / alphas_sum

    # start of recursion
    integrals[0, 0, 0, :, :, :] = np.sqrt(np.pi / alphas_sum) * np.exp(
        -harm_mean * (coord_a - coord_b) ** 2
    )

    # recurse over angular momentum for a
    # NOTE: array is sliced to avoid an if statement for angmom_a_max > 0
    integrals[0, 0, 1:2, :, :, :] = rel_coord_a * integrals[0, 0, 0:1, :, :, :]
    for i in range(1, angmom_a_max):
        integrals[0, 0, i + 1, :, :, :] = rel_coord_a * integrals[0, 0, i, :, :, :] + (
            i * integrals[0, 0, i - 1, :, :, :] / (2 * alphas_sum)
        )

    # recurse over angular momentum for b
    i_range = np.arange(angmom_a_max + 1)[None, None, :, None, None, None]
    # NOTE: array is sliced to avoid an if statement for angmom_b_max > 0
    integrals[0, 1:2, 0, :, :, :] = rel_coord_b * integrals[0, 0:1, 0, :, :, :]
    integrals[0, 1:2, 1:, :, :, :] = rel_coord_b * integrals[0, 0:1, 1:, :, :, :] + (
        i_range[0, 0:1, 1:, :, :, :] * integrals[0, 0:1, :-1, :, :, :] / (2 * alphas_sum)
    )
    for j in range(1, angmom_b_max):
        integrals[0, j + 1, 0, :, :, :] = rel_coord_b * integrals[0, j, 0, :, :, :] + (
            j * integrals[0, j - 1, 0, :, :, :] / (2 * alphas_sum)
        )
        integrals[0, j + 1, 1:, :, :, :] = rel_coord_b * integrals[0, j, 1:, :, :, :] + (
            i_range[0, 0, 1:, :, :, :] * integrals[0, j, :-1, :, :, :]
            + j * integrals[0, j - 1, 1:, :, :, :]
        ) / (2 * alphas_sum)
    # recurse over order of moment
    j_range = np.arange(angmom_b_max + 1)[None, :, None, None, None, None]
    # NOTE: array is sliced to avoid an if statement for angmom_b_max > 0
    integrals[1:2, 0, 0, :, :, :] = rel_coord_moment * integrals[0:1, 0, 0, :, :, :]
    integrals[1:2, 0, 1:, :, :, :] = rel_coord_moment * integrals[0:1, 0, 1:, :, :, :] + (
        i_range[0:1, 0, 1:, :, :, :] * integrals[0:1, 0, :-1, :, :, :] / (2 * alphas_sum)
    )
    integrals[1:2, 1:, 0, :, :, :] = rel_coord_moment * integrals[0:1, 1:, 0, :, :, :] + (
        j_range[0:1, 1:, 0, :, :, :] * integrals[0:1, :-1, 0, :, :, :] / (2 * alphas_sum)
    )
    integrals[1:2, 1:, 1:, :, :, :] = rel_coord_moment * integrals[0:1, 1:, 1:, :, :, :] + (
        i_range[0:1, 0:1, 1:, :, :, :] * integrals[0:1, 1:, :-1, :, :, :]
        + j_range[0:1, 1:, 0:1, :, :, :] * integrals[0:1, :-1, 1:, :, :, :]
    ) / (2 * alphas_sum)
    for k in range(1, order_moment_max):
        integrals[k + 1, 0, 0, :, :, :] = rel_coord_moment * integrals[k, 0, 0, :, :, :] + (
            k * integrals[k - 1, 0, 0, :, :, :] / (2 * alphas_sum)
        )
        integrals[k + 1, 0, 1:, :, :, :] = rel_coord_moment * integrals[k, 0, 1:, :, :, :] + (
            i_range[0, 0, 1:, :, :, :] * integrals[k, 0, :-1, :, :, :]
            + k * integrals[k - 1, 0, 1:, :, :, :]
        ) / (2 * alphas_sum)
        integrals[k + 1, 1:, 0, :, :, :] = rel_coord_moment * integrals[k, 1:, 0, :, :, :] + (
            j_range[0, 1:, 0, :, :, :] * integrals[k, :-1, 0, :, :, :]
            + k * integrals[k - 1, 1:, 0, :, :, :]
        ) / (2 * alphas_sum)
        integrals[k + 1, 1:, 1:, :, :, :] = rel_coord_moment * integrals[k, 1:, 1:, :, :, :] + (
            i_range[0, 0:, 1:, :, :, :] * integrals[k, 1:, :-1, :, :, :]
            + j_range[0, 1:, 0:, :, :, :] * integrals[k, :-1, 1:, :, :, :]
            + k * integrals[k - 1, 1:, 1:, :, :, :]
        ) / (2 * alphas_sum)

    # select the appropriate moments and angular momentums
    integrals = integrals[
        orders_moment[:, np.newaxis, np.newaxis, :],
        angmoms_b[np.newaxis, :, np.newaxis, :],
        angmoms_a[np.newaxis, np.newaxis, :, :],
        np.arange(3)[np.newaxis, np.newaxis, np.newaxis, :],
    ]

    # multiply the x, y, and z components together
    integrals = np.prod(integrals, axis=3)

    # transform the primitives
    # NOTE: axis for dimension (x, y, z) of the coordinate has been removed
    integrals = np.tensordot(integrals, coeffs_b, (4, 0))
    integrals = np.tensordot(integrals, coeffs_a, (3, 0))

    return integrals
