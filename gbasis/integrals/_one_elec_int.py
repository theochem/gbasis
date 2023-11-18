"""One-electron integrals involving Contracted Cartesian Gaussians."""
import numpy as np
from gbasis.utils import factorial2


# pylint: disable=C0103,R0914,R0915
# FIXME: returns nan when exponent is zero
def _compute_one_elec_integrals(
    coords_points,
    boys_func,
    coord_a,
    angmom_a,
    exps_a,
    coeffs_a,
    coord_b,
    angmom_b,
    exps_b,
    coeffs_b,
):
    r"""Return the one-electron integrals for a point charge interaction.

    Parameters
    ----------
    coords_points : np.ndarray(N, 3)
        Coordinates of the point charges.
    boys_func : function(orders, weighted_dist)
        Boys function used to evaluate the one-electron integral.
        `orders` is the orders of the Boys integral that will be evaluated. It should be a
        three-dimensional `numpy` array of integers with `shape` (M, 1, 1, 1) where
        `M` is the number of orders that will be evaluated.
        `weighted_dist` is the weighted interatomic distance, i.e.
        :math:`\frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{PC}||^2` where :math:`\alpha_i`
        is the exponent of the i-th primitive on the left side and the :math:`\beta_j` is the
        exponent of the j-th primitive on the right side. It should be a four-dimensional `numpy`
        array of floats with `shape` (1, N, K_b, K_a), where `N` is the number of
        point charges and `K_a` and `K_b` are the number of primitives on the left and
        right side, respectively.
        Output is the Boys function evaluated for each order and the weighted interatomic distance.
        It will be a three-dimensional `numpy` array with `shape` (M, N, K_b, K_a).
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmom_a : int
        Angular momentum of the segmented contractions on the left side.
        We will denote this value to be `L_a`.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmom_b : int
        Angular momentum of the segmented contractions on the right side.
        We will denote this value to be `L_b`.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives on the right side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).

    Returns
    -------
    integrals : np.ndarray(L_a + 1, L_a + 1, L_a + 1, L_b + 1, L_b + 1, L_b + 1, M_a, M_b)
        One electron integrals for the given `GeneralizedContractionShell` instances.
        Dimensions 0, 1, and 2 correspond to the :math:`x, y, \text{and} z` components of the
        angular momentum for contraction a.
        Dimensions 3, 4, and 5 correspond to the :math:`x, y, \text{and} z` components of the
        angular momentum for contraction b.
        Dimension 6 corresponds to the segmented contractions of contraction a.
        Dimension 7 corresponds to the segmented contractions of contraction b.

    """

    m_max = angmom_a + angmom_b + 1

    # NOTE: Ordering convention for vertical recursion of integrals
    # axis 0 : m (size: m_max)
    # axis 1 : a_x (size: m_max)
    # axis 2 : a_y (size: m_max)
    # axis 3 : a_z (size: m_max)
    # axis 4 : point charge (size: N)
    # axis 5 : primitive of contraction b (size: K_b)
    # axis 6 : primitive of contraction a (size: K_a)

    integrals = np.zeros(
        (m_max, m_max, m_max, m_max, coords_points.shape[0], exps_b.size, exps_a.size)
    )

    # Adjust axes for pre-work
    # axis 0 : m (size: m_max)
    # axis 1 : components of vectors (x, y, z) (size: 3)
    # axis 2 : point charge (size: N)
    # axis 3 : primitive of contraction b (size: K_b)
    # axis 4 : primitive of contraction a (size: K_a)
    coord_point = coords_points.T[np.newaxis, :, :, np.newaxis, np.newaxis]
    coord_a = coord_a[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    coord_b = coord_b[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    exps_a = exps_a[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    # sum of the exponents
    exps_sum = exps_a + exps_b
    # coordinate of the weighted average center
    coord_wac = (exps_a * coord_a + exps_b * coord_b) / exps_sum
    # relative distance from weighted average center
    rel_coord_a = coord_wac - coord_a  # R_pa
    rel_dist = coord_a - coord_b  # R_ab
    rel_coord_point = coord_wac - coord_point  # R_pc
    # harmonic mean
    harm_mean = exps_a * exps_b / exps_sum

    # Initialize V(m)(000|000) for all m
    integrals[:, 0, 0, 0, :, :, :] = (
        (2 * np.pi / exps_sum.squeeze(axis=1))
        * boys_func(
            np.arange(m_max)[:, None, None, None],
            (exps_sum.squeeze(axis=1) * np.sum(rel_coord_point ** 2, axis=1))[:, None, :, :],
        )
        * np.exp(-harm_mean.squeeze(axis=1) * (rel_dist ** 2).sum(axis=1))
    )

    # Vertical recursion for the first index
    integrals[:-1, 1:2, 0, 0, :, :, :] = (
        rel_coord_a[:, 0, :, :, :] * integrals[:-1, 0:1, 0, 0, :, :, :]
        - rel_coord_point[:, 0, :, :, :] * integrals[1:, 0:1, 0, 0, :, :, :]
    )
    for a in range(1, m_max - 1):
        integrals[:-1, a + 1, 0, 0, :, :, :] = (
            rel_coord_a[:, 0, :, :, :] * integrals[:-1, a, 0, 0, :, :, :]
            - rel_coord_point[:, 0, :, :, :] * integrals[1:, a, 0, 0, :, :, :]
            + a
            / (2 * exps_sum)[:, 0, :, :, :]
            * (integrals[:-1, a - 1, 0, 0, :, :, :] - integrals[1:, a - 1, 0, 0, :, :, :])
        )

    # Vertical recursion for the second index
    integrals[:-1, :, 1:2, 0, :, :, :] = (
        rel_coord_a[:, 1, :, :, :][:, None, :, :, :] * integrals[:-1, :, 0:1, 0, :, :, :]
        - rel_coord_point[:, 1, :, :, :][:, None, :, :, :] * integrals[1:, :, 0:1, 0, :, :, :]
    )
    for a in range(1, m_max - 1):
        integrals[:-1, :, a + 1, 0, :, :, :] = (
            rel_coord_a[:, 1, :, :, :][:, None, :, :, :] * integrals[:-1, :, a, 0, :, :, :]
            - rel_coord_point[:, 1, :, :, :][:, None, :, :, :] * integrals[1:, :, a, 0, :, :, :]
            + a
            / (2 * exps_sum).squeeze(axis=1)[:, None, :, :, :]  # a bit redundant, but okay
            * (integrals[:-1, :, a - 1, 0, :, :, :] - integrals[1:, :, a - 1, 0, :, :, :])
        )

    # Vertical recursion for the third index
    integrals[:-1, :, :, 1:2, :, :, :] = (
        rel_coord_a[:, 2, :, :, :][:, None, None, :, :, :] * integrals[:-1, :, :, 0:1, :, :, :]
        - rel_coord_point[:, 2, :, :, :][:, None, None, :, :, :] * integrals[1:, :, :, 0:1, :, :, :]
    )
    for a in range(1, m_max - 1):
        integrals[:-1, :, :, a + 1, :, :, :] = (
            rel_coord_a[:, 2, :, :, :][:, None, None, :, :, :] * integrals[:-1, :, :, a, :, :, :]
            - rel_coord_point[:, 2, :, :, :][:, None, None, :, :, :]
            * integrals[1:, :, :, a, :, :, :]
            + a
            / (2 * exps_sum).squeeze(axis=1)[:, None, None, :, :, :]  # a bit redundant, but okay
            * (integrals[:-1, :, :, a - 1, :, :, :] - integrals[1:, :, :, a - 1, :, :, :])
        )

    # Discard nonrelevant integrals
    integrals_cont = integrals[0, :, :, :, :, :, :]
    # Get normalization constants that correspond to the exponents (and the angular momentum)
    norm_a = (((2 * exps_a / np.pi) ** (3 / 4)) * ((4 * exps_a) ** (angmom_a / 2))).reshape(
        1, 1, 1, 1, 1, -1
    )
    norm_b = (((2 * exps_b / np.pi) ** (3 / 4)) * ((4 * exps_b) ** (angmom_b / 2))).reshape(
        1, 1, 1, 1, -1, 1
    )
    # Contract primitives
    integrals_cont = np.tensordot(integrals_cont * norm_a, coeffs_a, (5, 0))
    integrals_cont = np.tensordot(integrals_cont * norm_b, coeffs_b, (4, 0))

    # NOTE: Ordering convention for horizontal recursion of integrals
    # axis 0 : b_x (size: angmom_b + 1)
    # axis 1 : b_y (size: angmom_b + 1)
    # axis 2 : b_z (size: angmom_b + 1)
    # axis 3 : a_x (size: m_max)
    # axis 4 : a_y (size: m_max)
    # axis 5 : a_z (size: m_max)
    # axis 6 : point charge (size: N)
    # axis 7 : index for segmented contractions of contraction a (size: M_a)
    # axis 8 : index for segmented contractions of contraction b (size: M_b)
    integrals = np.zeros(
        (
            angmom_b + 1,
            angmom_b + 1,
            angmom_b + 1,
            m_max,
            m_max,
            m_max,
            coords_points.shape[0],
            coeffs_a.shape[1],
            coeffs_b.shape[1],
        )
    )
    rel_dist = np.squeeze(rel_dist)
    integrals[0, 0, 0, :, :, :, :, :, :] = integrals_cont

    # Horizontal recursion for the first index
    for b in range(0, angmom_b):
        # Increment b_x
        integrals[b + 1, 0, 0, :-1, :, :, :, :, :] = (
            integrals[b, 0, 0, 1:, :, :, :, :, :]
            + rel_dist[0] * integrals[b, 0, 0, :-1, :, :, :, :, :]
        )

    # Horizontal recursion for the second index
    for b in range(0, angmom_b):
        # Increment b_x
        integrals[:, b + 1, 0, :, :-1, :, :, :, :] = (
            integrals[:, b, 0, :, 1:, :, :, :, :]
            + rel_dist[1] * integrals[:, b, 0, :, :-1, :, :, :, :]
        )

    # Horizontal recursion for the third index
    for b in range(0, angmom_b):
        # Increment b_x
        integrals[:, :, b + 1, :, :, :-1, :, :, :] = (
            integrals[:, :, b, :, :, 1:, :, :, :]
            + rel_dist[2] * integrals[:, :, b, :, :, :-1, :, :, :]
        )

    # rearrange to more sensible order
    integrals = np.transpose(integrals, (3, 4, 5, 0, 1, 2, 6, 7, 8))

    # discard higher order angular momentum needed for the recursions
    integrals = integrals[: angmom_a + 1, : angmom_a + 1, : angmom_a + 1]

    # Get normalization constants that correspond to the angular momentum components
    angmoms_a = np.arange(angmom_a + 1)
    angmoms_b = np.arange(angmom_b + 1)
    norm_a = 1 / np.sqrt(
        factorial2(2 * angmoms_a[:, None, None, None, None, None, None, None, None] - 1)
        * factorial2(2 * angmoms_a[None, :, None, None, None, None, None, None, None] - 1)
        * factorial2(2 * angmoms_a[None, None, :, None, None, None, None, None, None] - 1)
    )
    norm_b = 1 / np.sqrt(
        factorial2(2 * angmoms_b[None, None, None, :, None, None, None, None, None] - 1)
        * factorial2(2 * angmoms_b[None, None, None, None, :, None, None, None, None] - 1)
        * factorial2(2 * angmoms_b[None, None, None, None, None, :, None, None, None] - 1)
    )
    integrals *= norm_a
    integrals *= norm_b

    return integrals
