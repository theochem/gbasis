"""One-electron integrals involving Contracted Cartesian Gaussians."""
import numpy as np
from scipy.special import factorial2


# pylint: disable=C0103,R0914,R0915
# FIXME: returns nan when exponent is zero
def _compute_one_elec_integrals(
    coord_point, boys_func, coord_a, angmom_a, exps_a, coeffs_a, coord_b, angmom_b, exps_b, coeffs_b
):
    r"""Return the one-electron integrals for a point charge interaction.

    Parameters
    ----------
    coord_point : np.ndarray(3,)
        Center of the point charge.
    boys_func : function(orders, weighted_dist)
        Boys function used to evaluate the one-electron integral.
        `orders` is the orders of the Boys integral that will be evaluated. It should be a
        three-dimensional numpy array of integers with shape `(M, 1, 1)` where `M` is the number of
        orders that will be evaluated.
        `weighted_dist` is the weighted interatomic distance, i.e.
        :math:`\frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{AB}||^2` where :math:`\alpha_i` is
        the exponent of the ith primitive on the left side and the :math:`\beta_j` is the exponent
        of the jth primitive on the right side. It should be a three-dimensional numpy array of
        floats with shape `(1, K_b, K_a)` where `K_a` and `K_b` are the number of primitives on the
        left and right side, respectively.
        Output is the Boys function evaluated for each order and the weighted interactomic distance.
        It will be a three-dimensional numpy array of shape `(M, K_b, K_a)`.
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
        where the first index corresponds to the primitive and the second index corresponds to the
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
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).

    Returns
    -------
    integrals : np.ndarray(L_a + 1, L_a + 1, L_a + 1, L_b + 1, L_b + 1, L_b + 1, M_a, M_b)
        One electron integrals for the given `ContractedCartesianGaussian` instances.
        First, second, and third index correspond to the `x`, `y`, and `z` components of the
        angular momentum for contraction a.
        Fourth, fifth, and sixth index correspond to the `x`, `y`, and `z` components of the
        angular momentum for contraction b.
        Seventh index corresponds to the segmented contractions of contraction a.
        Eighth index corresponds to the segmented contractions of contraction b.

    """

    m_max = angmom_a + angmom_b + 1

    # NOTE: Ordering convention for vertical recursion of integrals
    # axis 0 : m (size: m_max)
    # axis 1 : a_x (size: m_max)
    # axis 2 : a_y (size: m_max)
    # axis 3 : a_z (size: m_max)
    # axis 4 : primitive of contraction b (size: K_b)
    # axis 5 : primitive of contraction a (size: K_a)

    integrals = np.zeros((m_max, m_max, m_max, m_max, exps_b.size, exps_a.size))

    # Adjust axes for pre-work
    # axis 0 : m (size: m_max)
    # axis 1 : components of vectors (x, y, z) (size: 3)
    # axis 2 : primitive of contraction b (size: K_b)
    # axis 3 : primitive of contraction a (size: K_a)
    coord_point = coord_point[np.newaxis, :, np.newaxis, np.newaxis]
    coord_a = coord_a[np.newaxis, :, np.newaxis, np.newaxis]
    coord_b = coord_b[np.newaxis, :, np.newaxis, np.newaxis]
    exps_a = exps_a[np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, np.newaxis, :, np.newaxis]

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
    m = np.arange(m_max)[:, None, None]
    integrals[:, 0, 0, 0, :, :] = boys_func(
        m, exps_sum[:, 0] * np.sum(rel_coord_point ** 2, axis=1)
    ) * np.exp(-harm_mean[:, 0] * (rel_dist ** 2).sum(axis=1))

    # Vertical recursion for one nonzero index i.e. V(010|000)
    # For a = 0:
    integrals[:-1, [1, 0, 0], [0, 1, 0], [0, 0, 1], :, :] = (
        rel_coord_a * integrals[:-1, 0, 0, 0, :, :][:, None, :, :]
        - rel_coord_point * integrals[1:, 0, 0, 0, :, :][:, None, :, :]
    )
    # For a > 0:
    for a in range(1, m_max - 1):
        integrals[: -a - 1, [a + 1, 0, 0], [0, a + 1, 0], [0, 0, a + 1], :, :] = (
            rel_coord_a * integrals[: -a - 1, [a, 0, 0], [0, a, 0], [0, 0, a], :, :]
            - rel_coord_point * integrals[1:-a, [a, 0, 0], [0, a, 0], [0, 0, a], :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, [a - 1, 0, 0], [0, a - 1, 0], [0, 0, a - 1], :, :]
                - integrals[1:-a, [a - 1, 0, 0], [0, a - 1, 0], [0, 0, a - 1], :, :]
            )
        )

    # Vertical recursion for two nonzero indices i.e. V(110|000)
    # Slice to avoid if statement
    # For a = 0:
    # Increment a_x for all a_y:
    integrals[:-1, 1:2, 1:-1, 0, :, :] = (
        rel_coord_a[:, 0, :, :] * integrals[:-1, 0:1, 1:-1, 0, :, :]
        - rel_coord_point[:, 0, :, :] * integrals[1:, 0:1, 1:-1, 0, :, :]
    )
    # Increment a_x for all a_z:
    integrals[:-1, 1:2, 0, 1:-1, :, :] = (
        rel_coord_a[:, 0, :, :] * integrals[:-1, 0:1, 0, 1:-1, :, :]
        - rel_coord_point[:, 0, :, :] * integrals[1:, 0:1, 0, 1:-1, :, :]
    )
    # Increment a_y for all a_x:
    integrals[:-1, 1:-1, 1:2, 0, :, :] = (
        rel_coord_a[:, 1, :, :] * integrals[:-1, 1:-1, 0:1, 0, :, :]
        - rel_coord_point[:, 1, :, :] * integrals[1:, 1:-1, 0:1, 0, :, :]
    )
    # Increment a_y for all a_z
    integrals[:-1, 0, 1:2, 1:-1, :, :] = (
        rel_coord_a[:, 1, :, :] * integrals[:-1, 0, 0:1, 1:-1, :, :]
        - rel_coord_point[:, 1, :, :] * integrals[1:, 0, 0:1, 1:-1, :, :]
    )
    # Increment a_z for all a_x
    integrals[:-1, 1:-1, 0, 1:2, :, :] = (
        rel_coord_a[:, 2, :, :] * integrals[:-1, 1:-1, 0, 0:1, :, :]
        - rel_coord_point[:, 2, :, :] * integrals[1:, 1:-1, 0, 0:1, :, :]
    )
    # Increment a_z for all a_y
    integrals[:-1, 0, 1:-1, 1:2, :, :] = (
        rel_coord_a[:, 2, :, :] * integrals[:-1, 0, 1:-1, 0:1, :, :]
        - rel_coord_point[:, 2, :, :] * integrals[1:, 0, 1:-1, 0:1, :, :]
    )
    # For a > 0:
    for a in range(1, m_max - 1):
        # Increment a_x for all a_y:
        integrals[: -a - 1, a + 1, a + 1 : -a - 1, 0, :, :] = (
            rel_coord_a[:, 0, :, :] * integrals[: -a - 1, a, a + 1 : -a - 1, 0, :, :]
            - rel_coord_point[:, 0, :, :] * integrals[1:-a, a, a + 1 : -a - 1, 0, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a - 1, a + 1 : -a - 1, 0, :, :]
                - integrals[1:-a, a - 1, a + 1 : -a - 1, 0, :, :]
            )
        )
        # Increment a_x for all a_z:
        integrals[: -a - 1, a + 1, 0, a + 1 : -a - 1, :, :] = (
            rel_coord_a[:, 0, :, :] * integrals[: -a - 1, a, 0, a + 1 : -a - 1, :, :]
            - rel_coord_point[:, 0, :, :] * integrals[1:-a, a, 0, a + 1 : -a - 1, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a - 1, 0, a + 1 : -a - 1, :, :]
                - integrals[1:-a, a - 1, 0, a + 1 : -a - 1, :, :]
            )
        )
        # Increment a_y for all a_x:
        integrals[: -a - 1, a + 1 : -a - 1, a + 1, 0, :, :] = (
            rel_coord_a[:, 1, :, :] * integrals[: -a - 1, a + 1 : -a - 1, a, 0, :, :]
            - rel_coord_point[:, 1, :, :] * integrals[1:-a, a + 1 : -a - 1, a, 0, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a + 1 : -a - 1, a - 1, 0, :, :]
                - integrals[1:-a, a + 1 : -a - 1, a - 1, 0, :, :]
            )
        )
        # Increment a_y for all a_z
        integrals[: -a - 1, 0, a + 1, a + 1 : -a - 1, :, :] = (
            rel_coord_a[:, 1, :, :] * integrals[: -a - 1, 0, a, a + 1 : -a - 1, :, :]
            - rel_coord_point[:, 1, :, :] * integrals[1:-a, 0, a, a + 1 : -a - 1, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, 0, a - 1, a + 1 : -a - 1, :, :]
                - integrals[1:-a, 0, a - 1, a + 1 : -a - 1, :, :]
            )
        )
        # Increment a_z for all a_x
        integrals[: -a - 1, a + 1 : -a - 1, 0, a + 1, :, :] = (
            rel_coord_a[:, 2, :, :] * integrals[: -a - 1, a + 1 : -a - 1, 0, a, :, :]
            - rel_coord_point[:, 2, :, :] * integrals[1:-a, a + 1 : -a - 1, 0, a, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a + 1 : -a - 1, 0, a - 1, :, :]
                - integrals[1:-a, a + 1 : -a - 1, 0, a - 1, :, :]
            )
        )
        # Increment a_z for all a_y
        integrals[: -a - 1, 0, a + 1 : -a - 1, a + 1, :, :] = (
            rel_coord_a[:, 2, :, :] * integrals[: -a - 1, 0, a + 1 : -a - 1, a, :, :]
            - rel_coord_point[:, 2, :, :] * integrals[1:-a, 0, a + 1 : -a - 1, a, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, 0, a + 1 : -a - 1, a - 1, :, :]
                - integrals[1:-a, 0, a + 1 : -a - 1, a - 1, :, :]
            )
        )

    # Vertical recursion for three nonzero indices i.e. V(111|000)
    # Slice to avoid if statement
    # For a = 0:
    integrals[:-2, 1:2, 1:-1, 1:-1, :, :] = (
        rel_coord_a[:, 0, :, :] * integrals[:-2, 0:1, 1:-1, 1:-1, :, :]
        - rel_coord_point[:, 0, :, :] * integrals[1:-1, 0:1, 1:-1, 1:-1, :, :]
    )
    integrals[:-2, 1:-1, 1:2, 1:-1, :, :] = (
        rel_coord_a[:, 1, :, :] * integrals[:-2, 1:-1, 0:1, 1:-1, :, :]
        - rel_coord_point[:, 1, :, :] * integrals[1:-1, 1:-1, 0:1, 1:-1, :, :]
    )
    integrals[:-2, 1:-1, 1:-1, 1:2, :, :] = (
        rel_coord_a[:, 2, :, :] * integrals[:-2, 1:-1, 1:-1, 0:1, :, :]
        - rel_coord_point[:, 2, :, :] * integrals[1:-1, 1:-1, 1:-1, 0:1, :, :]
    )
    # For a > 0:
    for a in range(1, m_max - 1):
        # Increment a_x for all a_y, a_z:
        integrals[: -a - 1, a + 1, a + 1 : -a - 1, a + 1 : -a - 1, :, :] = (
            rel_coord_a[:, 0, :, :] * integrals[: -a - 1, a, a + 1 : -a - 1, a + 1 : -a - 1, :, :]
            - rel_coord_point[:, 0, :, :] * integrals[1:-a, a, a + 1 : -a - 1, a + 1 : -a - 1, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a - 1, a + 1 : -a - 1, a + 1 : -a - 1, :, :]
                - integrals[1:-a, a - 1, a + 1 : -a - 1, a + 1 : -a - 1, :, :]
            )
        )
        # Increment a_y for all a_x, a_z:
        integrals[: -a - 1, a + 1 : -a - 1, a + 1, a + 1 : -a - 1, :, :] = (
            rel_coord_a[:, 0, :, :] * integrals[: -a - 1, a + 1 : -a - 1, a, a + 1 : -a - 1, :, :]
            - rel_coord_point[:, 0, :, :] * integrals[1:-a, a + 1 : -a - 1, a, a + 1 : -a - 1, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a + 1 : -a - 1, a - 1, a + 1 : -a - 1, :, :]
                - integrals[1:-a, a + 1 : -a - 1, a - 1, a + 1 : -a - 1, :, :]
            )
        )
        # Increment a_z for all a_x, a_y:
        integrals[: -a - 1, a + 1 : -a - 1, a + 1 : -a - 1, a + 1, :, :] = (
            rel_coord_a[:, 0, :, :] * integrals[: -a - 1, a + 1 : -a - 1, a + 1 : -a - 1, a, :, :]
            - rel_coord_point[:, 0, :, :] * integrals[1:-a, a + 1 : -a - 1, a + 1 : -a - 1, a, :, :]
            + a
            / (2 * exps_sum)
            * (
                integrals[: -a - 1, a + 1 : -a - 1, a + 1 : -a - 1, a - 1, :, :]
                - integrals[1:-a, a + 1 : -a - 1, a + 1 : -a - 1, a - 1, :, :]
            )
        )

    # Discard nonrelevant integrals
    integrals_cont = integrals[0, :, :, :, :, :]
    # Get norms
    # FIXME: hard coded
    exps_a_norm = exps_a.reshape(1, 1, 1, -1)
    angmoms_a_x = np.arange(m_max)[:, None, None, None]
    angmoms_a_y = np.arange(m_max)[None, :, None, None]
    angmoms_a_z = np.arange(m_max)[None, None, :, None]
    norm_a = (
        (2 * exps_a_norm / np.pi) ** (3 / 4)
        * ((4 * exps_a) ** ((angmoms_a_x + angmoms_a_y + angmoms_a_z) / 2))
        / np.sqrt(
            factorial2(2 * angmoms_a_x - 1)
            * factorial2(2 * angmoms_a_y - 1)
            * factorial2(2 * angmoms_a_z - 1)
        )
    )[:, :, :, None, :]
    norm_b = ((2 * exps_b / np.pi) ** (3 / 4)).reshape(1, 1, 1, 1, -1, 1)
    # Contract primitives
    integrals_cont = np.tensordot(integrals_cont * norm_a, coeffs_a, (4, 0))
    integrals_cont = np.tensordot(integrals_cont * norm_b, coeffs_b, (3, 0))

    # NOTE: Ordering convention for horizontal recursion of integrals
    # axis 0 : b_x (size: angmom_b + 1)
    # axis 1 : b_y (size: angmom_b + 1)
    # axis 2 : b_z (size: angmom_b + 1)
    # axis 3 : a_x (size: m_max)
    # axis 4 : a_y (size: m_max)
    # axis 5 : a_z (size: m_max)
    # axis 6 : index for segmented contractions of contraction a (size: M_a)
    # axis 7 : index for segmented contractions of contraction b (size: M_b)
    integrals = np.zeros(
        (
            angmom_b + 1,
            angmom_b + 1,
            angmom_b + 1,
            m_max,
            m_max,
            m_max,
            coeffs_a.shape[1],
            coeffs_b.shape[1],
        )
    )
    rel_dist = np.squeeze(rel_dist)
    integrals[0, 0, 0, :, :, :, :, :] = integrals_cont

    # Horizontal recursion for one nonzero index i.e. V(120|100)
    for b in range(0, angmom_b):
        # Increment b_x
        integrals[b + 1, 0, 0, :-1, :, :, :, :] = (
            integrals[b, 0, 0, 1:, :, :, :, :] + rel_dist[0] * integrals[b, 0, 0, :-1, :, :, :, :]
        )
        # Increment b_y
        integrals[0, b + 1, 0, :, :-1, :, :, :] = (
            integrals[0, b, 0, :, 1:, :, :, :] + rel_dist[1] * integrals[0, b, 0, :, :-1, :, :, :]
        )
        # Increment b_z
        integrals[0, 0, b + 1, :, :, :-1, :, :] = (
            integrals[0, 0, b, :, :, 1:, :, :] + rel_dist[2] * integrals[0, 0, b, :, :, :-1, :, :]
        )

    # Horizontal recursion for two nonzero indices
    for b in range(0, angmom_b):
        # Increment b_x for all b_y
        integrals[b + 1, b + 1 : -b - 1, 0, :-1, :, :, :, :] = (
            integrals[b, b + 1 : -b - 1, 0, 1:, :, :, :, :]
            + rel_dist[0] * integrals[b, b + 1 : -b - 1, 0, :-1, :, :, :, :]
        )
        # Increment b_x for all b_z
        integrals[b + 1, 0, b + 1 : -b - 1, :-1, :, :, :, :] = (
            integrals[b, 0, b + 1 : -b - 1, 1:, :, :, :, :]
            + rel_dist[0] * integrals[b, 0, b + 1 : -b - 1, :-1, :, :, :, :]
        )
        # Increment b_y for all b_x
        integrals[b + 1 : -b - 1, b + 1, 0, :, :-1, :, :, :] = (
            integrals[b + 1 : -b - 1, b, 0, :, 1:, :, :, :]
            + rel_dist[1] * integrals[b + 1 : -b - 1, b, 0, :, :-1, :, :, :]
        )
        # Increment b_y for all b_z
        integrals[0, b + 1, b + 1 : -b - 1, :, :-1, :, :, :] = (
            integrals[0, b, b + 1 : -b - 1, :, 1:, :, :, :]
            + rel_dist[1] * integrals[0, b, b + 1 : -b - 1, :, :-1, :, :, :]
        )
        # Increment b_z for all b_x
        integrals[b + 1 : -b - 1, 0, b + 1, :, :, :-1, :, :] = (
            integrals[b + 1 : -b - 1, 0, b, :, :, 1:, :, :]
            + rel_dist[2] * integrals[b + 1 : -b - 1, 0, b, :, :, :-1, :, :]
        )
        # Increment b_z for all b_y
        integrals[0, b + 1 : -b - 1, b + 1, :, :, :-1, :, :] = (
            integrals[0, b + 1 : -b - 1, b, :, :, 1:, :, :]
            + rel_dist[2] * integrals[0, b + 1 : -b - 1, b, :, :, :-1, :, :]
        )

    # Horizontal recursion for three nonzero indices
    for b in range(0, angmom_b):
        integrals[b + 1, b + 1 : -b - 1, b + 1 : -b - 1, :-2, :, :, :, :] = (
            integrals[b, b + 1 : -b - 1, b + 1 : -b - 1, 1:-1, :, :, :, :]
            + rel_dist[0] * integrals[b, b + 1 : -b - 1, b + 1 : -b - 1, :-2, :, :, :, :]
        )
        integrals[b + 1 : -b - 1, b + 1, b + 1 : -b - 1, :-2, :, :, :, :] = (
            integrals[b + 1 : -b - 1, b, b + 1 : -b - 1, 1:-1, :, :, :, :]
            + rel_dist[1] * integrals[b + 1 : -b - 1, b, b + 1 : -b - 1, :-2, :, :, :, :]
        )
        integrals[b + 1 : -b - 1, b + 1 : -b - 1, b + 1, :-2, :, :, :, :] = (
            integrals[b + 1 : -b - 1, b + 1 : -b - 1, b, 1:-1, :, :, :, :]
            + rel_dist[2] * integrals[b + 1 : -b - 1, b + 1 : -b - 1, b, :-2, :, :, :, :]
        )

    # rearrange to more sensible order
    integrals = np.transpose(integrals, (3, 4, 5, 0, 1, 2, 6, 7))

    # discard higher order angular momentum needed for the recursions
    integrals = integrals[: angmom_a + 1, : angmom_a + 1, : angmom_a + 1]

    return integrals
