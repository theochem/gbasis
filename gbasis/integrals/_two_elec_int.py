"""Two-electron integrals involving Contracted Cartesian Gaussians."""
import numpy as np

from gbasis.utils import factorial2

# pylint: disable=C0103,R0914,R0915


def _compute_two_elec_integrals_angmom_zero(
    boys_func,
    coord_a,
    exps_a,
    coeffs_a,
    coord_b,
    exps_b,
    coeffs_b,
    coord_c,
    exps_c,
    coeffs_c,
    coord_d,
    exps_d,
    coeffs_d,
):
    r"""Return the two-electron integrals for electron-electron repulsion for the s orbital case.

    All of the contractions are assumed to have angular momentum zero.

    .. math::

        \int \int \phi^*_a(\mathbf{r}_1) \phi_b(\mathbf{r}_1) g(\mathbf{r}_1 - \mathbf{r}_2)
        \phi^*_c(\mathbf{r}_2) \phi_d(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    boys_func : function(orders, weighted_dist)
        Boys function used to evaluate the two-electron integral.
        `orders` is the orders of the Boys integral that will be evaluated. It should be a
        three-dimensional `numpy` array of integers with `shape` (M, 1, 1, 1) where :
        math:`M` is the number of orders that will be evaluated.
        `weighted_dist` is the weighted interatomic distance, i.e.
        :math:`\frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{PC}||^2` where :math:`\alpha_i` is
        the exponent of the i-th primitive on the left side and the :math:`\beta_j` is the exponent
        of the j-th primitive on the right side. It should be a four-dimensional `numpy` array of
        floats with `shape` (1, N, K_b, K_a) where `N` is the number of point charges
        and `K_a` and `K_b` are the number of primitives on the left and right side,
        respectively.
        Output is the Boys function evaluated for each order and the weighted interatomic distance.
        It will be a three-dimensional `numpy` array with `shape` (M, N, K_b, K_a).
    coord_a : np.ndarray(3,)
        Center of the contraction a.
    exps_a : np.ndarray(K_a,)
        Exponents of the primitives in contraction a.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives in contraction a.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_b : np.ndarray(3,)
        Center of the contraction b.
    exps_b : np.ndarray(K_b,)
        Exponents of the primitives in contraction b.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives in contraction b.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_c : np.ndarray(3,)
        Center of the contraction c.
    exps_c : np.ndarray(K_c,)
        Exponents of the primitives in contraction c.
    coeffs_c : np.ndarray(K_c, M_c)
        Contraction coefficients of the primitives in contraction c.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_d : np.ndarray(3,)
        Center of the contraction d.
    exps_d : np.ndarray(K_d,)
        Exponents of the primitives in contraction d.
    coeffs_d : np.ndarray(K_d, M_d)
        Contraction coefficients of the primitives in contraction d.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).

    Returns
    -------
    integrals : np.ndarray(1, 1, 1, 1, M_a, M_b, M_c, M_d)
        Two-electron integrals **in Chemists' notation** of the four generalized contraction shells
        (a, b, c, d).
        Dimension 0 correspond to angular momentum components of contraction a.
        Dimension 1 correspond to angular momentum components of contraction b.
        Dimension 2 correspond to angular momentum components of contraction c.
        Dimension 3 correspond to angular momentum components of contraction d.
        Dimension 4 corresponds to the segmented contractions of contraction a.
        Dimension 5 corresponds to the segmented contractions of contraction b.
        Dimension 6 corresponds to the segmented contractions of contraction c.
        Dimension 7 corresponds to the segmented contractions of contraction d.
        Note that the integrals are in Chemists' notation.

    Notes
    -----
    All of the contractions are assumed to have angular momentum of zero.
    Integrals are in Chemists' notation.

    """
    coord_a = coord_a[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_b = coord_b[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_c = coord_c[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_d = coord_d[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    exps_a = exps_a[np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, :, np.newaxis, np.newaxis]
    exps_c = exps_c[np.newaxis, np.newaxis, :, np.newaxis]
    exps_d = exps_d[:, np.newaxis, np.newaxis, np.newaxis]

    harm_mean_one = (exps_a * exps_b) / (exps_a + exps_b)
    harm_mean_two = (exps_c * exps_d) / (exps_c + exps_d)
    harm_mean = (exps_a + exps_b) * (exps_c + exps_d) / (exps_a + exps_b + exps_c + exps_d)

    coord_wac = (exps_a * coord_a + exps_b * coord_b) / (exps_a + exps_b) - (
        exps_c * coord_c + exps_d * coord_d
    ) / (exps_c + exps_d)

    integrals = (
        (2 * np.pi**2.5)
        / ((exps_a + exps_b) * (exps_c + exps_d) * (exps_a + exps_b + exps_c + exps_d) ** 0.5)
        * boys_func(0, (harm_mean * np.sum(coord_wac**2, axis=0))[:, :, :, :])
        * np.exp(-harm_mean_one * np.sum((coord_a - coord_b) ** 2, axis=0))
        * np.exp(-harm_mean_two * np.sum((coord_c - coord_d) ** 2, axis=0))
    )

    norm_a = ((2 * exps_a / np.pi) ** (3 / 4)).reshape(1, 1, 1, -1)
    integrals = np.tensordot(integrals * norm_a, coeffs_a, (3, 0))

    norm_c = ((2 * exps_c / np.pi) ** (3 / 4)).reshape(1, 1, -1, 1)
    integrals = np.tensordot(integrals * norm_c, coeffs_c, (2, 0))

    norm_b = ((2 * exps_b / np.pi) ** (3 / 4)).reshape(1, -1, 1, 1)
    integrals = np.tensordot(integrals * norm_b, coeffs_b, (1, 0))

    norm_d = ((2 * exps_d / np.pi) ** (3 / 4)).reshape(-1, 1, 1, 1)
    integrals = np.tensordot(integrals * norm_d, coeffs_d, (0, 0))

    integrals = np.transpose(integrals, (0, 2, 1, 3))
    return integrals[None, None, None, None]


# FIXME: returns nan when exponent is zero
def _compute_two_elec_integrals(
    boys_func,
    coord_a,
    angmom_a,
    angmom_components_a,
    exps_a,
    coeffs_a,
    coord_b,
    angmom_b,
    angmom_components_b,
    exps_b,
    coeffs_b,
    coord_c,
    angmom_c,
    angmom_components_c,
    exps_c,
    coeffs_c,
    coord_d,
    angmom_d,
    angmom_components_d,
    exps_d,
    coeffs_d,
):
    r"""Return the two-electron integrals for electron-electron repulsion.

    .. math::

        \int \int \phi^*_a(\mathbf{r}_1) \phi_b(\mathbf{r}_1) g(\mathbf{r}_1 - \mathbf{r}_2)
        \phi^*_c(\mathbf{r}_2) \phi_d(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    boys_func : function(orders, weighted_dist)
        Boys function used to evaluate the two-electron integral.
        `orders` is the orders of the Boys integral that will be evaluated. It should be a
        three-dimensional `numpy` array of integers with `shape` (M, 1, 1, 1) where
        `M` is the number of orders that will be evaluated.
        `weighted_dist` is the weighted interatomic distance, i.e.
        :math:`\frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{PC}||^2` where :math:`\alpha_i` is
        the exponent of the i-th primitive on the left side and the :math:`\beta_j` is the exponent
        of the j-th primitive on the right side. It should be a four-dimensional `numpy` array of
        floats with `shape` (1, N, K_b, K_a) where `N` is the number of point charges
        and `K_a` and `K_b` are the number of primitives on the left and right side,
        respectively.
        Output is the Boys function evaluated for each order and the weighted interatomic distance.
        It will be a three-dimensional `numpy` array with `shape` (M, N, K_b, K_a).
    coord_a : np.ndarray(3,)
        Center of the contraction a.
    angmom_a : int
        Angular momentum of the contraction a.
    angmom_components_a : int
        Angular momentum components of the contraction a.
        All of the angular momentum components (in the x, y, and z) must add up to the angular
        momentum.
    exps_a : np.ndarray(K_a,)
        Exponents of the primitives in contraction a.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives in contraction a.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_b : np.ndarray(3,)
        Center of the contraction b.
    angmom_b : int
        Angular momentum of the contraction b.
    angmom_components_b : int
        Angular momentum components of the contraction b.
        All of the angular momentum components (in the x, y, and z) must add up to the angular
        momentum.
    exps_b : np.ndarray(K_b,)
        Exponents of the primitives in contraction b.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives in contraction b.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_c : np.ndarray(3,)
        Center of the contraction c.
    angmom_c : int
        Angular momentum of the contraction c.
    angmom_components_c : int
        Angular momentum components of the contraction c.
        All of the angular momentum components (in the x, y, and z) must add up to the angular
        momentum.
    exps_c : np.ndarray(K_c,)
        Exponents of the primitives in contraction c.
    coeffs_c : np.ndarray(K_c, M_c)
        Contraction coefficients of the primitives in contraction c.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).
    coord_d : np.ndarray(3,)
        Center of the contraction d.
    angmom_d : int
        Angular momentum of the contraction d.
    angmom_components_d : int
        Angular momentum components of the contraction d.
        All of the angular momentum components (in the x, y, and z) must add up to the angular
        momentum.
    exps_d : np.ndarray(K_d,)
        Exponents of the primitives in contraction d.
    coeffs_d : np.ndarray(K_d, M_d)
        Contraction coefficients of the primitives in contraction d.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where dimension 0 corresponds to the primitive and dimension 1 corresponds to the
        contraction (with the same exponents and angular momentum).

    Returns
    -------
    integrals : np.ndarray(L_a, L_b, L_c, L_d, M_a, M_b, M_c, M_d)
        Two-electron integrals **in Chemists' notation** of the four generalized contraction shells
        (a, b, c, d).
        Dimension 0 correspond to angular momentum components of contraction a.
        Dimension 1 correspond to angular momentum components of contraction b.
        Dimension 2 correspond to angular momentum components of contraction c.
        Dimension 3 correspond to angular momentum components of contraction d.
        Dimension 4 corresponds to the segmented contractions of contraction a.
        Dimension 5 corresponds to the segmented contractions of contraction b.
        Dimension 6 corresponds to the segmented contractions of contraction c.
        Dimension 7 corresponds to the segmented contractions of contraction d.
        Note that the integrals are in Chemists' notation.

    Raises
    ------
    ValueError
        If angular momentum of all four of the contractions (a, b, c, d) are zero.

    Notes
    -----
    The case where all four of the contractions have angular momentum of zero is not supported. Use
    `gbasis.integrals._two_elec_int._compute_two_elec_integrals` instead.

    """

    m_max = angmom_a + angmom_b + angmom_c + angmom_d + 1
    m_max_a = angmom_a + angmom_b + 1
    m_max_c = angmom_c + angmom_d + 1

    # Adjust axes for pre-work
    # axis 0 : components of vectors (x, y, z) (size: 3)
    # axis 1 : primitive of contraction d (size: K_d)
    # axis 2 : primitive of contraction b (size: K_b)
    # axis 3 : primitive of contraction c (size: K_c)
    # axis 4 : primitive of contraction a (size: K_a)
    coord_a = coord_a[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_b = coord_b[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_c = coord_c[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_d = coord_d[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    # axis 0 : primitive of contraction d (size: K_d)
    # axis 1 : primitive of contraction b (size: K_b)
    # axis 2 : primitive of contraction c (size: K_c)
    # axis 3 : primitive of contraction a (size: K_a)
    exps_a = exps_a[np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, :, np.newaxis, np.newaxis]
    exps_c = exps_c[np.newaxis, np.newaxis, :, np.newaxis]
    exps_d = exps_d[:, np.newaxis, np.newaxis, np.newaxis]

    # sum of the exponents
    exps_sum_one = exps_a + exps_b
    exps_sum_two = exps_c + exps_d
    exps_sum = exps_sum_one + exps_sum_two
    # harmonic mean
    harm_mean_one = (exps_a * exps_b) / exps_sum_one
    harm_mean_two = (exps_c * exps_d) / exps_sum_two
    harm_mean = (exps_a + exps_b) * (exps_c + exps_d) / exps_sum
    # coordinate of the weighted average center
    coord_wac_one = (exps_a * coord_a + exps_b * coord_b) / exps_sum_one
    coord_wac_two = (exps_c * coord_c + exps_d * coord_d) / exps_sum_two
    coord_wac = coord_wac_one - coord_wac_two
    # relative distance from weighted average center
    rel_dist_one = coord_a - coord_b
    rel_dist_two = coord_c - coord_d
    rel_coord_a = coord_wac_one - coord_a
    rel_coord_c = coord_wac_two - coord_c

    # NOTE: Ordering convention for vertical recursion of integrals
    # axis 0 : m (size: m_max)
    # axis 1 : a_x (size: m_max)
    # axis 2 : a_y (size: m_max)
    # axis 3 : a_z (size: m_max)
    # axis 4 : primitive of contraction d (size: K_d)
    # axis 5 : primitive of contraction b (size: K_b)
    # axis 6 : primitive of contraction c (size: K_c)
    # axis 7 : primitive of contraction a (size: K_a)

    # FIXME: memory heavy here
    integrals_vert = np.zeros(
        (m_max, m_max, m_max, m_max, exps_d.size, exps_b.size, exps_c.size, exps_a.size)
    )
    # Initialize V(m)(000|000) for all m
    integrals_vert[:, 0, 0, 0, :, :, :, :] = (
        (2 * np.pi**2.5)
        / (exps_sum_one * exps_sum_two * exps_sum**0.5)
        * boys_func(
            np.arange(m_max)[:, None, None, None, None],
            (harm_mean * np.sum(coord_wac**2, axis=0))[None, :, :, :, :],
        )
        * np.exp(-harm_mean_one * np.sum(rel_dist_one**2, axis=0))
        * np.exp(-harm_mean_two * np.sum(rel_dist_two**2, axis=0))
    )

    # Vertical recursion for the first index
    integrals_vert[:-1, 1:2, 0, 0, :, :, :, :] = (
        rel_coord_a[0] * integrals_vert[:-1, 0:1, 0, 0]
        - harm_mean / exps_sum_one * coord_wac[0] * integrals_vert[1:, 0:1, 0, 0]
    )
    for a in range(1, m_max - 1):
        integrals_vert[:-1, a + 1, 0, 0, :, :, :, :] = (
            rel_coord_a[0] * integrals_vert[:-1, a, 0, 0]
            - harm_mean / exps_sum_one * coord_wac[0] * integrals_vert[1:, a, 0, 0]
            + a
            / (2 * exps_sum_one)
            * (
                integrals_vert[:-1, a - 1, 0, 0]
                - harm_mean / exps_sum_one * integrals_vert[1:, a - 1, 0, 0]
            )
        )

    # Vertical recursion for the second index
    integrals_vert[:-1, :, 1:2, 0, :, :, :, :] = (
        rel_coord_a[1] * integrals_vert[:-1, :, 0:1, 0]
        - harm_mean / exps_sum_one * coord_wac[1] * integrals_vert[1:, :, 0:1, 0]
    )
    for a in range(1, m_max - 1):
        integrals_vert[:-1, :, a + 1, 0, :, :, :, :] = (
            rel_coord_a[1] * integrals_vert[:-1, :, a, 0]
            - harm_mean / exps_sum_one * coord_wac[1] * integrals_vert[1:, :, a, 0]
            + a
            / (2 * exps_sum_one)
            * (
                integrals_vert[:-1, :, a - 1, 0]
                - harm_mean / exps_sum_one * integrals_vert[1:, :, a - 1, 0]
            )
        )

    # Vertical recursion for the third index
    integrals_vert[:-1, :, :, 1:2, :, :, :, :] = (
        rel_coord_a[2] * integrals_vert[:-1, :, :, 0:1]
        - harm_mean / exps_sum_one * coord_wac[2] * integrals_vert[1:, :, :, 0:1]
    )
    for a in range(1, m_max - 1):
        integrals_vert[:-1, :, :, a + 1, :, :, :, :] = (
            rel_coord_a[2] * integrals_vert[:-1, :, :, a]
            - harm_mean / exps_sum_one * coord_wac[2] * integrals_vert[1:, :, :, a]
            + a
            / (2 * exps_sum_one)
            * (
                integrals_vert[:-1, :, :, a - 1]
                - harm_mean / exps_sum_one * integrals_vert[1:, :, :, a - 1]
            )
        )

    # FIXME: memory heavy here
    # NOTE: Ordering convention for electron transfer recursion
    # axis 0 : c_x (size: m_max_c)
    # axis 1 : c_y (size: m_max_c)
    # axis 2 : c_z (size: m_max_c)
    # axis 3 : a_x (size: m_max)
    # axis 4 : a_y (size: m_max)
    # axis 5 : a_z (size: m_max)
    # axis 6 : primitive of contraction d (size: K_d)
    # axis 7 : primitive of contraction b (size: K_b)
    # axis 8 : primitive of contraction c (size: K_c)
    # axis 9 : primitive of contraction a (size: K_a)
    integrals_etransf = np.zeros(
        (
            m_max_c,
            m_max_c,
            m_max_c,
            m_max,
            m_max,
            m_max,
            exps_d.size,
            exps_b.size,
            exps_c.size,
            exps_a.size,
        )
    )
    # Discard m values
    integrals_etransf[0, 0, 0, :, :, :, :, :, :, :] = integrals_vert[0, :, :, :, :, :, :, :]
    # TODO: check if actually discarded

    # electron transfer recursion for first index
    # NOTE: At this point, numpy (v1.16.3) does not support broadcasting two zero size arrays of
    # different shapes (e.g. shape(0, 1) and shape(1, 0))into one another. The slices 1:2 on the 0th
    # and the 3rd axis causes problems when they both have dimension zero. There doesn't seem to be
    # an easy enough way around it, so the breaking case (all contractions have angular momentum of
    # 0) is not supported. Use `_compute_two_elec_integrals_angmom_zero` for this case.
    # FIXME: i couldn't get the base case (all angmom of zero) to work.
    integrals_etransf[1:2, 0, 0, 0:1, :, :, :, :, :, :] = (
        rel_coord_c[0] + exps_sum_one / exps_sum_two * rel_coord_a[0]
    ) * integrals_etransf[0:1, 0, 0, 0:1] - exps_sum_one / exps_sum_two * integrals_etransf[
        0:1, 0, 0, 1:2
    ]
    integrals_etransf[1:2, 0, 0, 1:-1, :, :, :, :, :, :] = (
        (rel_coord_c[0] + exps_sum_one / exps_sum_two * rel_coord_a[0])
        * integrals_etransf[0:1, 0, 0, 1:-1]
        + np.arange(1, m_max - 1).reshape(1, 1, 1, -1, 1, 1, 1, 1, 1, 1)
        / (2 * exps_sum_two)
        * integrals_etransf[0:1, 0, 0, :-2]
        - exps_sum_one / exps_sum_two * integrals_etransf[0:1:, 0, 0, 2:]
    )
    for c in range(1, m_max_c - 1):
        integrals_etransf[c + 1, 0, 0, 0, :, :, :, :, :, :] = (
            (rel_coord_c[0] + exps_sum_one / exps_sum_two * rel_coord_a[0])
            * integrals_etransf[c, 0, 0, 0]
            + c / (2 * exps_sum_two) * integrals_etransf[c - 1, 0, 0, 0]
            - exps_sum_one / exps_sum_two * integrals_etransf[c, 0, 0, 1]
        )
        integrals_etransf[c + 1, 0, 0, 1:-1, :, :, :, :, :, :] = (
            (rel_coord_c[0] + exps_sum_one / exps_sum_two * rel_coord_a[0])
            * integrals_etransf[c, 0, 0, 1:-1]
            + np.arange(1, m_max - 1).reshape(1, 1, 1, -1, 1, 1, 1, 1, 1, 1)
            / (2 * exps_sum_two)
            * integrals_etransf[c, 0, 0, :-2]
            + c / (2 * exps_sum_two) * integrals_etransf[c - 1, 0, 0, 1:-1]
            - exps_sum_one / exps_sum_two * integrals_etransf[c, 0, 0, 2:]
        )
    # electron transfer recursion for second index
    integrals_etransf[:, 1:2, 0, :, 0:1, :, :, :, :, :] = (
        rel_coord_c[1] + exps_sum_one / exps_sum_two * rel_coord_a[1]
    ) * integrals_etransf[:, 0:1, 0, :, 0:1] - exps_sum_one / exps_sum_two * integrals_etransf[
        :, 0:1, 0, :, 1:2
    ]
    integrals_etransf[:, 1:2, 0, :, 1:-1, :, :, :, :, :] = (
        (rel_coord_c[1] + exps_sum_one / exps_sum_two * rel_coord_a[1])
        * integrals_etransf[:, 0:1, 0, :, 1:-1]
        + np.arange(1, m_max - 1).reshape(1, 1, 1, 1, -1, 1, 1, 1, 1, 1)
        / (2 * exps_sum_two)
        * integrals_etransf[:, 0:1, 0, :, :-2]
        - exps_sum_one / exps_sum_two * integrals_etransf[:, 0:1, 0, :, 2:]
    )
    for c in range(1, m_max_c - 1):
        integrals_etransf[:, c + 1, 0, :, 0, :, :, :, :, :] = (
            (rel_coord_c[1] + exps_sum_one / exps_sum_two * rel_coord_a[1])
            * integrals_etransf[:, c, 0, :, 0]
            + c / (2 * exps_sum_two) * integrals_etransf[:, c - 1, 0, :, 0]
            - exps_sum_one / exps_sum_two * integrals_etransf[:, c, 0, :, 1]
        )
        integrals_etransf[:, c + 1, 0, :, 1:-1, :, :, :, :, :] = (
            (rel_coord_c[1] + exps_sum_one / exps_sum_two * rel_coord_a[1])
            * integrals_etransf[:, c, 0, :, 1:-1]
            + np.arange(1, m_max - 1).reshape(1, 1, 1, 1, -1, 1, 1, 1, 1, 1)
            / (2 * exps_sum_two)
            * integrals_etransf[:, c, 0, :, :-2]
            + c / (2 * exps_sum_two) * integrals_etransf[:, c - 1, 0, :, 1:-1]
            - exps_sum_one / exps_sum_two * integrals_etransf[:, c, 0, :, 2:]
        )
    # electron transfer recursion for third index
    integrals_etransf[:, :, 1:2, :, :, 0:1, :, :, :, :] = (
        rel_coord_c[2] + exps_sum_one / exps_sum_two * rel_coord_a[2]
    ) * integrals_etransf[:, :, 0:1, :, :, 0:1] - exps_sum_one / exps_sum_two * integrals_etransf[
        :, :, 0:1, :, :, 1:2
    ]
    integrals_etransf[:, :, 1:2, :, :, 1:-1, :, :, :, :] = (
        (rel_coord_c[2] + exps_sum_one / exps_sum_two * rel_coord_a[2])
        * integrals_etransf[:, :, 0:1, :, :, 1:-1]
        + np.arange(1, m_max - 1).reshape(1, 1, 1, 1, 1, -1, 1, 1, 1, 1)
        / (2 * exps_sum_two)
        * integrals_etransf[:, :, 0:1, :, :, :-2]
        - exps_sum_one / exps_sum_two * integrals_etransf[:, :, 0:1, :, :, 2:]
    )
    for c in range(1, m_max_c - 1):
        integrals_etransf[:, :, c + 1, :, :, 0, :, :, :, :] = (
            (rel_coord_c[2] + exps_sum_one / exps_sum_two * rel_coord_a[2])
            * integrals_etransf[:, :, c, :, :, 0]
            + c / (2 * exps_sum_two) * integrals_etransf[:, :, c - 1, :, :, 0]
            - exps_sum_one / exps_sum_two * integrals_etransf[:, :, c, :, :, 1]
        )
        integrals_etransf[:, :, c + 1, :, :, 1:-1, :, :, :, :] = (
            (rel_coord_c[2] + exps_sum_one / exps_sum_two * rel_coord_a[2])
            * integrals_etransf[:, :, c, :, :, 1:-1]
            + np.arange(1, m_max - 1).reshape(1, 1, 1, 1, 1, -1, 1, 1, 1, 1)
            / (2 * exps_sum_two)
            * integrals_etransf[:, :, c, :, :, :-2]
            + c / (2 * exps_sum_two) * integrals_etransf[:, :, c - 1, :, :, 1:-1]
            - exps_sum_one / exps_sum_two * integrals_etransf[:, :, c, :, :, 2:]
        )

    # Contract primitives (after normalizing)
    norm_a = (((2 * exps_a / np.pi) ** (3 / 4)) * ((4 * exps_a) ** (angmom_a / 2))).reshape(
        1, 1, 1, 1, 1, 1, 1, 1, 1, -1
    )
    integrals_cont = np.tensordot(integrals_etransf * norm_a, coeffs_a, (9, 0))

    norm_c = (((2 * exps_c / np.pi) ** (3 / 4)) * ((4 * exps_c) ** (angmom_c / 2))).reshape(
        1, 1, 1, 1, 1, 1, 1, 1, -1, 1
    )
    integrals_cont = np.tensordot(integrals_cont * norm_c, coeffs_c, (8, 0))

    norm_b = (((2 * exps_b / np.pi) ** (3 / 4)) * ((4 * exps_b) ** (angmom_b / 2))).reshape(
        1, 1, 1, 1, 1, 1, 1, -1, 1, 1
    )
    integrals_cont = np.tensordot(integrals_cont * norm_b, coeffs_b, (7, 0))

    norm_d = (((2 * exps_d / np.pi) ** (3 / 4)) * ((4 * exps_d) ** (angmom_d / 2))).reshape(
        1, 1, 1, 1, 1, 1, -1, 1, 1, 1
    )
    integrals_cont = np.tensordot(integrals_cont * norm_d, coeffs_d, (6, 0))

    # NOTE: Ordering convention for horizontal recursion of first and second indices of d
    # axis 0 : d_x (size: angmom_d + 1)
    # axis 1 : d_y (size: angmom_d + 1)
    # axis 2 : c_x (size: m_max_c)
    # axis 3 : c_y (size: m_max_c)
    # axis 4 : c_z (size: m_max_c)
    # axis 5 : a_x (size: m_max_a)
    # axis 6 : a_y (size: m_max_a)
    # axis 7 : a_z (size: m_max_a)
    # axis 8 : segmented contraction a (size: M_a)
    # axis 9 : segmented contraction c (size: M_c)
    # axis 10 : segmented contraction b (size: M_b)
    # axis 11 : segmented contraction d (size: M_d)
    # FIXME: memory heavy here
    # NOTE: For higher angular momentum (> 6), the third index makes pushes the memory usage over
    # a limit (to tens of GB of memory for a single primitive), so we only recurse the first two
    # indices, then select the desired angular momentum components, then recurse over the third
    # index.
    integrals_horiz_d = np.zeros(
        (
            angmom_d + 1,
            angmom_d + 1,
            m_max_c,
            m_max_c,
            m_max_c,
            m_max_a,
            m_max_a,
            m_max_a,
            coeffs_a.shape[1],
            coeffs_c.shape[1],
            coeffs_b.shape[1],
            coeffs_d.shape[1],
        )
    )
    integrals_horiz_d[0, 0, :, :, :, :, :, :, :, :, :, :] = integrals_cont[
        :, :, :, :m_max_a, :m_max_a, :m_max_a, :, :, :, :
    ]
    # TODO: check if actually discarded

    # FIXME: rearrange indices for slightly better indexing (i.e. order z, y, x)
    # Horizontal recursion for first index of d
    for d in range(0, angmom_d):
        integrals_horiz_d[d + 1, 0, :-1, :, :, :, :, :, :, :, :, :] = (
            integrals_horiz_d[d, 0, 1:, :, :, :, :, :, :, :, :, :]
            + rel_dist_two[0] * integrals_horiz_d[d, 0, :-1, :, :, :, :, :, :, :, :, :]
        )
    # Horizontal recursion for the second index of d
    for d in range(0, angmom_d):
        integrals_horiz_d[:, d + 1, :, :-1, :, :, :, :, :, :, :, :] = (
            integrals_horiz_d[:, d, :, 1:, :, :, :, :, :, :, :, :]
            + rel_dist_two[1] * integrals_horiz_d[:, d, :, :-1, :, :, :, :, :, :, :, :]
        )

    # NOTE: we select the angular momentum components of the recursed indices (x and y) here instead
    # at the end in order to construct a smaller (intermediate) array thereby decreasing memory
    # usage
    # NOTE: Ordering convention for horizontal recursion of third index of d
    # axis 0 : d_z (size: angmom_d + 1)
    # axis 1 : c_z (size: m_max_c)
    # axis 2 : angular momentum component of d (size: L_d)
    # axis 3 : angular momentum component of c (size: L_c)
    # axis 4 : a_x (size: m_max_a)
    # axis 5 : a_y (size: m_max_a)
    # axis 6 : a_z (size: m_max_a)
    # axis 7 : segmented contraction a (size: M_a)
    # axis 8 : segmented contraction c (size: M_c)
    # axis 9 : segmented contraction b (size: M_b)
    # axis 10 : segmented contraction d (size: M_d)
    integrals_horiz_d2 = np.zeros(
        (
            angmom_d + 1,
            m_max_c,
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            m_max_a,
            m_max_a,
            m_max_a,
            coeffs_a.shape[1],
            coeffs_c.shape[1],
            coeffs_b.shape[1],
            coeffs_d.shape[1],
        )
    )

    # remove unneeded angular momentum components (x and y)
    angmoms_c_x, angmoms_c_y, angmoms_c_z = angmom_components_c.T
    angmoms_d_x, angmoms_d_y, angmoms_d_z = angmom_components_d.T
    integrals_horiz_d2[0] = np.transpose(
        integrals_horiz_d[
            angmoms_d_x.reshape(-1, 1),
            angmoms_d_y.reshape(-1, 1),
            angmoms_c_x.reshape(1, -1),
            angmoms_c_y.reshape(1, -1),
            :,
            :,
            :,
            :,
            :,
            :,
            :,
            :,
        ],
        (2, 0, 1, 3, 4, 5, 6, 7, 8, 9),
    )

    # Horizontal recursion for the third index of d
    for d in range(0, angmom_d):
        integrals_horiz_d2[d + 1, :-1, :, :, :, :, :, :, :, :, :] = (
            integrals_horiz_d2[d, 1:, :, :, :, :, :, :, :, :, :]
            + rel_dist_two[2] * integrals_horiz_d2[d, :-1, :, :, :, :, :, :, :, :, :]
        )

    # remove unneeded angular momentum components (z)
    integrals_horiz_d2 = integrals_horiz_d2[
        angmoms_d_z.reshape(-1, 1),
        angmoms_c_z.reshape(1, -1),
        np.arange(angmoms_d_z.size).reshape(-1, 1),
        np.arange(angmoms_c_z.size).reshape(1, -1),
        :,
        :,
        :,
        :,
        :,
        :,
        :,
    ]

    # NOTE: Ordering convention for horizontal recursion of first and second indices of b
    # axis 0 : b_x (size: angmom_b + 1)
    # axis 1 : b_y (size: angmom_b + 1)
    # axis 2 : a_x (size: m_max_a)
    # axis 3 : a_y (size: m_max_a)
    # axis 4 : a_z (size: m_max_a)
    # axis 5 : angular momentum component of d (size: L_d)
    # axis 6 : angular momentum component of c (size: L_c)
    # axis 7 : segmented contraction a (size: M_a)
    # axis 8 : segmented contraction c (size: M_c)
    # axis 9 : segmented contraction b (size: M_b)
    # axis 10 : segmented contraction d (size: M_d)
    integrals_horiz_b = np.zeros(
        (
            angmom_b + 1,
            angmom_b + 1,
            m_max_a,
            m_max_a,
            m_max_a,
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            coeffs_a.shape[1],
            coeffs_c.shape[1],
            coeffs_b.shape[1],
            coeffs_d.shape[1],
        )
    )
    integrals_horiz_b[0, 0, :, :, :, :, :, :, :, :, :] = np.transpose(
        integrals_horiz_d2[:, :, :, :, :, :, :, :, :], (2, 3, 4, 0, 1, 5, 6, 7, 8)
    )
    # TODO: check if actually discarded
    # Horizontal recursion for first index of b
    for b in range(0, angmom_b):
        integrals_horiz_b[b + 1, 0, :-1, :, :, :, :, :, :, :, :] = (
            integrals_horiz_b[b, 0, 1:, :, :, :, :, :, :, :, :]
            + rel_dist_one[0] * integrals_horiz_b[b, 0, :-1, :, :, :, :, :, :, :, :]
        )
    # Horizontal recursion for the second index of b
    for b in range(0, angmom_b):
        integrals_horiz_b[:, b + 1, :, :-1, :, :, :, :, :, :, :] = (
            integrals_horiz_b[:, b, :, 1:, :, :, :, :, :, :, :]
            + rel_dist_one[1] * integrals_horiz_b[:, b, :, :-1, :, :, :, :, :, :, :]
        )

    # NOTE: we select the angular momentum components of the recursed indices (x and y) here instead
    # at the end in order to construct a smaller (intermediate) array thereby decreasing memory
    # usage
    # NOTE: Ordering convention for horizontal recursion of third index of b
    # axis 0 : b_z (size: angmom_b + 1)
    # axis 1 : a_z (size: m_max_a)
    # axis 2 : angular momentum component of b (size: L_b)
    # axis 3 : angular momentum component of a (size: L_a)
    # axis 4 : angular momentum component of d (size: L_d)
    # axis 5 : angular momentum component of c (size: L_c)
    # axis 6 : segmented contraction a (size: M_a)
    # axis 7 : segmented contraction c (size: M_c)
    # axis 8 : segmented contraction b (size: M_b)
    # axis 9 : segmented contraction d (size: M_d)
    integrals_horiz_b2 = np.zeros(
        (
            angmom_b + 1,
            m_max_a,
            angmom_components_b.shape[0],
            angmom_components_a.shape[0],
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            coeffs_a.shape[1],
            coeffs_c.shape[1],
            coeffs_b.shape[1],
            coeffs_d.shape[1],
        )
    )

    # remove unneeded angular momentum components (x and y)
    angmoms_a_x, angmoms_a_y, angmoms_a_z = angmom_components_a.T
    angmoms_b_x, angmoms_b_y, angmoms_b_z = angmom_components_b.T
    integrals_horiz_b2[0] = np.transpose(
        integrals_horiz_b[
            angmoms_b_x.reshape(-1, 1),
            angmoms_b_y.reshape(-1, 1),
            angmoms_a_x.reshape(1, -1),
            angmoms_a_y.reshape(1, -1),
            :,
            :,
            :,
            :,
            :,
            :,
            :,
        ],
        (2, 0, 1, 3, 4, 5, 6, 7, 8),
    )

    # Horizontal recursion for the third index of b
    for b in range(0, angmom_b):
        integrals_horiz_b2[b + 1, :-1, :, :, :, :, :, :, :, :] = (
            integrals_horiz_b2[b, 1:, :, :, :, :, :, :, :, :]
            + rel_dist_one[2] * integrals_horiz_b2[b, :-1, :, :, :, :, :, :, :, :]
        )

    # remove unneeded angular momentum components (z)
    integrals_horiz_b2 = integrals_horiz_b2[
        angmoms_b_z.reshape(-1, 1),
        angmoms_a_z.reshape(1, -1),
        np.arange(angmoms_b_z.size).reshape(-1, 1),
        np.arange(angmoms_a_z.size).reshape(1, -1),
        :,
        :,
        :,
        :,
        :,
        :,
    ]

    # rearrange to more sensible order
    integrals = np.transpose(integrals_horiz_b2, (1, 0, 3, 2, 4, 6, 5, 7))

    # Get normalization constants that correspond to the angular momentum components
    norm_a = 1 / np.sqrt(np.prod(factorial2(2 * angmom_components_a - 1), axis=1)).reshape(
        -1, 1, 1, 1, 1, 1, 1, 1
    )
    integrals *= norm_a

    norm_b = 1 / np.sqrt(np.prod(factorial2(2 * angmom_components_b - 1), axis=1)).reshape(
        1, -1, 1, 1, 1, 1, 1, 1
    )
    integrals *= norm_b

    norm_c = 1 / np.sqrt(np.prod(factorial2(2 * angmom_components_c - 1), axis=1)).reshape(
        1, 1, -1, 1, 1, 1, 1, 1
    )
    integrals *= norm_c

    norm_d = 1 / np.sqrt(np.prod(factorial2(2 * angmom_components_d - 1), axis=1)).reshape(
        1, 1, 1, -1, 1, 1, 1, 1
    )
    integrals *= norm_d

    return integrals
