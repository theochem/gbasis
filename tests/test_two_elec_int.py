"""Test gbasis.integrals._two_elec_int."""
import numpy as np
import pytest
from scipy.special import hyp1f1  # pylint: disable=E0611

from gbasis.integrals._two_elec_int import (
    _compute_two_elec_integrals,
    _compute_two_elec_integrals_angmom_zero,
)
from gbasis.utils import factorial2


def boys_func(order, weighted_dist):
    r"""Boys function for evaluating the one-electron Coulomb interaction integral.

    The Coulombic Boys function can be written as a renormalized special case of the Kummer
    confluent hypergeometric function, as derived in Helgaker (eq. 9.8.39).

    Parameters
    ----------
    order : int
        Differentiation order of the helper function.
        Same as m in eq. 23, Aldrichs, R. Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
    weighted_dist : np.ndarray(L_b, L_a)
        The weighted interatomic distance, :math:`\\mu * R_{AB}^{2}`, where `\\mu` is the
        harmonic mean of the exponents for primitive one and two,
        :math:`\\mu = \\alpha * \\beta / (\\alpha + \\beta)`.
        `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

    Returns
    -------
    boys_eval : np.ndarray(L_b, L_a)
        The evaluation of the Boys function for the given values. This output corresponds
        to the evaluation for every Gaussian primitive at the given differentiation order.

    Notes
    -----
    There's some documented instability for hyp1f1, mainly for large values or complex numbers.
    In this case it seems fine, since m should be less than 10 in most cases, and except for
    exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
    problem becomes a precision error in most cases where it was an overflow error before, so
    the values should be close even when they are wrong.
    This function cannot be vectorized for both m and x.

    """

    return hyp1f1(order + 1 / 2, order + 3 / 2, -weighted_dist) / (2 * order + 1)


def two_int_brute(
    i_0,
    i_1,
    i_2,
    j_0,
    j_1,
    j_2,
    k_0,
    k_1,
    k_2,
    l_0,
    l_1,
    l_2,
    m,
    alpha=0.1,
    beta=0.2,
    gamma=0.3,
    delta=0.4,
    coord_a=(0.2, 0.4, 0.6),
    coord_b=(1.0, 1.5, 2.0),
    coord_c=(0.1, 0.3, 0.5),
    coord_d=(1.1, 1.6, 2.1),
    output=False,
):
    """Return the answer to the two-electron integral tests.

    Data for first primitive on the left:
    - Coordinate: [0.2, 0.4, 0.6]
    - Coefficients: [1.0]

    Data for first primitive on the right:
    - Coordinate: [1.0, 1.5, 2.0]
    - Coefficients: [1.0]

    Data for second primitive on the left:
    - Coordinate: [0.1, 0.3, 0.5]
    - Coefficients: [1.0]

    Data for second primitive on the right:
    - Coordinate: [1.1, 1.6, 2.1]
    - Coefficients: [1.0]


    Parameters
    ----------
    i_0 : int
        Angular momentum component (x) for the given coordinate of the first primitive on the left.
    i_1 : int
        Angular momentum component (y) for the given coordinate of the first primitive on the left.
    i_2 : int
        Angular momentum component (z) for the given coordinate of the first primitive on the left.
    j_0 : int
        Angular momentum component (x) for the given coordinate of the first primitive on the right.
    j_1 : int
        Angular momentum component (y) for the given coordinate of the first primitive on the right.
    j_2 : int
        Angular momentum component (z) for the given coordinate of the first primitive on the right.
    k_0 : int
        Angular momentum component (x) for the given coordinate of the second primitive on the left.
    k_1 : int
        Angular momentum component (y) for the given coordinate of the second primitive on the left.
    k_2 : int
        Angular momentum component (z) for the given coordinate of the second primitive on the left.
    l_0 : int
        Angular momentum component (x) for the given coordinate of the second primitive on the
        right.
    l_1 : int
        Angular momentum component (y) for the given coordinate of the second primitive on the
        right.
    l_2 : int
        Angular momentum component (z) for the given coordinate of the second primitive on the
        right.

    Returns
    -------
    answer : float

    """
    zeta = alpha + beta
    eta = gamma + delta
    rho = zeta * eta / (zeta + eta)
    x_a, y_a, z_a = coord_a
    x_b, y_b, z_b = coord_b
    x_c, y_c, z_c = coord_c
    x_d, y_d, z_d = coord_d

    x_p = (alpha * x_a + beta * x_b) / zeta
    y_p = (alpha * y_a + beta * y_b) / zeta
    z_p = (alpha * z_a + beta * z_b) / zeta
    x_q = (gamma * x_c + delta * x_d) / eta
    y_q = (gamma * y_c + delta * y_d) / eta
    z_q = (gamma * z_c + delta * z_d) / eta

    olp_ab = np.exp(-alpha * beta / zeta * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2))
    olp_cd = np.exp(-gamma * delta / eta * ((x_c - x_d) ** 2 + (y_c - y_d) ** 2 + (z_c - z_d) ** 2))

    if (
        i_0 < 0
        or i_1 < 0
        or i_2 < 0
        or j_0 < 0
        or j_1 < 0
        or j_2 < 0
        or k_0 < 0
        or k_1 < 0
        or k_2 < 0
        or l_0 < 0
        or l_1 < 0
        or l_2 < 0
    ):
        return 0.0

    if output:
        norm_a = (
            (2 * alpha / np.pi) ** (3 / 4)
            * ((4 * alpha) ** ((i_0 + i_1 + i_2) / 2.0))
            / ((factorial2(2 * i_0 - 1) * factorial2(2 * i_1 - 1) * factorial2(2 * i_2 - 1)) ** 0.5)
        )
        norm_b = (
            (2 * beta / np.pi) ** (3 / 4)
            * ((4 * beta) ** ((j_0 + j_1 + j_2) / 2))
            / ((factorial2(2 * j_0 - 1) * factorial2(2 * j_1 - 1) * factorial2(2 * j_2 - 1)) ** 0.5)
        )
        norm_c = (
            (2 * gamma / np.pi) ** (3 / 4)
            * ((4 * gamma) ** ((k_0 + k_1 + k_2) / 2))
            / ((factorial2(2 * k_0 - 1) * factorial2(2 * k_1 - 1) * factorial2(2 * k_2 - 1)) ** 0.5)
        )
        norm_d = (
            (2 * delta / np.pi) ** (3 / 4)
            * ((4 * delta) ** ((l_0 + l_1 + l_2) / 2))
            / ((factorial2(2 * l_0 - 1) * factorial2(2 * l_1 - 1) * factorial2(2 * l_2 - 1)) ** 0.5)
        )
        norm = norm_a * norm_b * norm_c * norm_d
    else:
        norm = 1

    # initial
    if i_0 == i_1 == i_2 == j_0 == j_1 == j_2 == k_0 == k_1 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            (2 * np.pi**2.5)
            / (zeta * eta * (zeta + eta) ** 0.5)
            * boys_func(m, (rho) * ((x_p - x_q) ** 2 + (y_p - y_q) ** 2 + (z_p - z_q) ** 2))
            * olp_ab
            * olp_cd
            * norm
        )
    # vertical recursion
    if i_1 == i_2 == j_0 == j_1 == j_2 == k_0 == k_1 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            (x_p - x_a)
            * two_int_brute(
                i_0 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - rho
            / zeta
            * (x_p - x_q)
            * two_int_brute(
                i_0 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m + 1,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (i_0 - 1)
            / 2
            / zeta
            * (
                two_int_brute(
                    i_0 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
                - rho
                / zeta
                * two_int_brute(
                    i_0 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m + 1,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
            )
        ) * norm
    if i_2 == j_0 == j_1 == j_2 == k_0 == k_1 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            (y_p - y_a)
            * two_int_brute(
                i_0,
                i_1 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - rho
            / zeta
            * (y_p - y_q)
            * two_int_brute(
                i_0,
                i_1 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m + 1,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (i_1 - 1)
            / 2
            / zeta
            * (
                two_int_brute(
                    i_0,
                    i_1 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
                - rho
                / zeta
                * two_int_brute(
                    i_0,
                    i_1 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m + 1,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
            )
        ) * norm
    if j_0 == j_1 == j_2 == k_0 == k_1 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            (z_p - z_a)
            * two_int_brute(
                i_0,
                i_1,
                i_2 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - rho
            / zeta
            * (z_p - z_q)
            * two_int_brute(
                i_0,
                i_1,
                i_2 - 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                m + 1,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (i_2 - 1)
            / 2
            / zeta
            * (
                two_int_brute(
                    i_0,
                    i_1,
                    i_2 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
                - rho
                / zeta
                * two_int_brute(
                    i_0,
                    i_1,
                    i_2 - 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    m + 1,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    coord_a,
                    coord_b,
                    coord_c,
                    coord_d,
                )
            )
        ) * norm
    # electron transfer
    if j_0 == j_1 == j_2 == k_1 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            ((x_q - x_c) + zeta / eta * (x_p - x_a))
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0 - 1,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + i_0
            / 2
            / eta
            * two_int_brute(
                i_0 - 1,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0 - 1,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (k_0 - 1)
            / 2
            / eta
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0 - 2,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - zeta
            / eta
            * two_int_brute(
                i_0 + 1,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0 - 1,
                0,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    if j_0 == j_1 == j_2 == k_2 == l_0 == l_1 == l_2 == 0:
        return (
            ((y_q - y_c) + zeta / eta * (y_p - y_a))
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0,
                k_1 - 1,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + i_1
            / 2
            / eta
            * two_int_brute(
                i_0,
                i_1 - 1,
                i_2,
                0,
                0,
                0,
                k_0,
                k_1 - 1,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (k_1 - 1)
            / 2
            / eta
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0,
                k_1 - 2,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - zeta
            / eta
            * two_int_brute(
                i_0,
                i_1 + 1,
                i_2,
                0,
                0,
                0,
                k_0,
                k_1 - 1,
                0,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    if j_0 == j_1 == j_2 == l_0 == l_1 == l_2 == 0:
        return (
            ((z_q - z_c) + zeta / eta * (z_p - z_a))
            * two_int_brute(
                i_0, i_1, i_2, 0, 0, 0, k_0, k_1, k_2 - 1, 0, 0, 0, m, alpha, beta, gamma, delta
            )
            + i_2
            / 2
            / eta
            * two_int_brute(
                i_0,
                i_1,
                i_2 - 1,
                0,
                0,
                0,
                k_0,
                k_1,
                k_2 - 1,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (k_2 - 1)
            / 2
            / eta
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                0,
                0,
                0,
                k_0,
                k_1,
                k_2 - 2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            - zeta
            / eta
            * two_int_brute(
                i_0,
                i_1,
                i_2 + 1,
                0,
                0,
                0,
                k_0,
                k_1,
                k_2 - 1,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    # horizontal recursion for b
    if j_1 == j_2 == l_0 == l_1 == l_2 == 0:
        return (
            two_int_brute(
                i_0 + 1,
                i_1,
                i_2,
                j_0 - 1,
                0,
                0,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (x_a - x_b)
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0 - 1,
                j_1,
                j_2,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    if j_2 == l_0 == l_1 == l_2 == 0:
        return (
            two_int_brute(
                i_0,
                i_1 + 1,
                i_2,
                j_0,
                j_1 - 1,
                0,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (y_a - y_b)
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1 - 1,
                j_2,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    if l_0 == l_1 == l_2 == 0:
        return (
            two_int_brute(
                i_0,
                i_1,
                i_2 + 1,
                j_0,
                j_1,
                j_2 - 1,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (z_a - z_b)
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1,
                j_2 - 1,
                k_0,
                k_1,
                k_2,
                0,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    # horizontal recursion for d
    if l_1 == l_2 == 0:
        return (
            two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1,
                j_2,
                k_0 + 1,
                k_1,
                k_2,
                l_0 - 1,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (x_c - x_d)
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1,
                j_2,
                k_0,
                k_1,
                k_2,
                l_0 - 1,
                0,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    if l_2 == 0:
        return (
            two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1,
                j_2,
                k_0,
                k_1 + 1,
                k_2,
                l_0,
                l_1 - 1,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
            + (y_c - y_d)
            * two_int_brute(
                i_0,
                i_1,
                i_2,
                j_0,
                j_1,
                j_2,
                k_0,
                k_1,
                k_2,
                l_0,
                l_1 - 1,
                0,
                m,
                alpha,
                beta,
                gamma,
                delta,
                coord_a,
                coord_b,
                coord_c,
                coord_d,
            )
        ) * norm
    return (
        two_int_brute(
            i_0,
            i_1,
            i_2,
            j_0,
            j_1,
            j_2,
            k_0,
            k_1,
            k_2 + 1,
            l_0,
            l_1,
            l_2 - 1,
            m,
            alpha,
            beta,
            gamma,
            delta,
            coord_a,
            coord_b,
            coord_c,
            coord_d,
        )
        + (z_c - z_d)
        * two_int_brute(
            i_0,
            i_1,
            i_2,
            j_0,
            j_1,
            j_2,
            k_0,
            k_1,
            k_2,
            l_0,
            l_1,
            l_2 - 1,
            m,
            alpha,
            beta,
            gamma,
            delta,
            coord_a,
            coord_b,
            coord_c,
            coord_d,
        )
    ) * norm


def test_two_int_brute():
    """Test two_int_brute by comparing it to HORTON's results.

    A basis set was created with the given specifications

        Data for first primitive on the left:
        - Coordinate: [0.2, 0.4, 0.6]
        - Exponents: [0.1]
        - Coefficients: [1.0]

        Data for first primitive on the right:
        - Coordinate: [1.0, 1.5, 2.0]
        - Exponents: [0.2]
        - Coefficients: [1.0]

        Data for second primitive on the left:
        - Coordinate: [0.1, 0.3, 0.5]
        - Exponents: [0.3]
        - Coefficients: [1.0]

        Data for second primitive on the right:
        - Coordinate: [1.1, 1.6, 2.1]
        - Exponents: [0.4]
        - Coefficients: [1.0]

    with the given angular momentum. Then the two-electron integrals were obtained, appropriate
    pieces were used as a reference. Following code was used to generate the two-electron integrals:

    ```python
    from horton import *
    import numpy as np

    mol = IOData.from_file('example.xyz')
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'ano-rcc', pure=False)

    lf = DenseLinalgFactory(obasis.nbasis)

    er = obasis.compute_electron_repulsion(lf)
    np.save("../../gbasis/tests/data_horton_test_initial.npy", er._array)
    ```

    """
    # initial
    #  s orbitals
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.1467279087841229
    )

    # vertical recursion
    #  p orbitals
    assert np.allclose(
        two_int_brute(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.048154271426258756
    )
    assert np.allclose(
        two_int_brute(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.06609629465618197
    )
    assert np.allclose(
        two_int_brute(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.08403831788610522
    )
    #  d orbitals
    assert np.allclose(
        two_int_brute(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.05244715697793453
    )
    assert np.allclose(
        two_int_brute(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.0217062830727606
    )
    assert np.allclose(
        two_int_brute(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.027599148189803257
    )
    assert np.allclose(
        two_int_brute(0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.060519542518842465
    )
    assert np.allclose(
        two_int_brute(0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.03788411662894913
    )
    assert np.allclose(
        two_int_brute(0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), 0.07112771009507537
    )

    # electron transfer recursion
    #  s & p orbitals
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True), 0.0928406204255668
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, output=True), 0.12085330330691318
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, output=True), 0.14886598618825944
    )
    #  s & d orbitals
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, output=True), 0.09927994323113691
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, output=True), 0.07647660304843684
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, output=True), 0.09420383743511908
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, output=True), 0.1228379603032861
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, output=True), 0.12262949926414721
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, output=True), 0.15257271733917935
    )
    #  p & p orbitals
    assert np.allclose(
        two_int_brute(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True), 0.047379713835400634
    )
    assert np.allclose(
        two_int_brute(1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, output=True), 0.039651931284181034
    )
    assert np.allclose(
        two_int_brute(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, output=True), 0.04884199202229442
    )
    assert np.allclose(
        two_int_brute(0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True), 0.04181115399876404
    )
    assert np.allclose(
        two_int_brute(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, output=True), 0.07134292715508984
    )
    assert np.allclose(
        two_int_brute(0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, output=True), 0.06703901373274967
    )
    assert np.allclose(
        two_int_brute(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True), 0.053160437451460436
    )
    assert np.allclose(
        two_int_brute(0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, output=True), 0.06919823644733271
    )
    assert np.allclose(
        two_int_brute(0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, output=True), 0.10215387873253794
    )

    # horizontal recursion for b
    #  p orbital
    assert np.allclose(
        two_int_brute(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True), -0.03688952129884961
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, output=True), -0.05088689809920155
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, output=True), -0.0648842748995535
    )
    # horizontal recursion for d
    #  p orbital
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, output=True), -0.07839464083963653
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, output=True), -0.1017277074036494
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, output=True), -0.12506077396766221
    )

    # different exponents
    assert np.allclose(
        two_int_brute(
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
            0,
            output=True,
            alpha=0.2,
            beta=0.3,
            gamma=0.4,
            delta=0.5,
        ),
        -0.015147214477383354,
    )

    # sspp
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True), 0.09284062042556687
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, output=True), -0.07839464083963657
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, output=True), 0.0811247584910971
    )
    # ssdd
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, output=True), 0.09927994323113697
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, output=True), 0.11133452322791097
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, output=True), 0.09517166304851872
    )
    # ssff
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, output=True), 0.06503259723193466
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, output=True), -0.06833176852396329
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, output=True), 0.0875150768518511
    )
    # sspp diff exponets
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.011285298359112576,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        -0.01128529835911257,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        -3.724520910598359e-05,
    )
    # ssdd diff exponets
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.01300963655953441,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.013009636559534408,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.007474093219268355,
    )
    # ssff diff exponets
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.01162655601223461,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        -0.011626556012234605,
    )
    assert np.allclose(
        two_int_brute(0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0.01, 0.01, 1, 1, output=True),
        0.0059448648293465006,
    )


def test_compute_two_elec_integrals_angmom_zero_prim():
    """Test gbasis.integrals._two_elec_int._compute_two_elec_integrals_angmom_zero on primitives."""
    assert np.allclose(
        _compute_two_elec_integrals_angmom_zero(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
    )
    assert np.allclose(
        _compute_two_elec_integrals_angmom_zero(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            np.array([0.1, 0.15]),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 1.5, 2.0]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, alpha=0.1)
        + 2 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, alpha=0.15),
    )
    assert np.allclose(
        _compute_two_elec_integrals_angmom_zero(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            np.array([0.2, 0.25]),
            np.array([[1.0], [2.0]]),
            np.array([0.1, 0.3, 0.5]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, beta=0.2)
        + 2 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, beta=0.25),
    )
    assert np.allclose(
        _compute_two_elec_integrals_angmom_zero(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            np.array([0.3, 0.35]),
            np.array([[1.0], [2.0]]),
            np.array([1.1, 1.6, 2.1]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, gamma=0.3)
        + 2 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, gamma=0.35),
    )
    assert np.allclose(
        _compute_two_elec_integrals_angmom_zero(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            np.array([0.4, 0.45]),
            np.array([[1.0], [2.0]]),
        ),
        1 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, delta=0.4)
        + 2 * two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True, delta=0.45),
    )


def test_compute_two_elec_integrals_prim():
    """Test gbasis.integrals._two_elec_int._compute_two_elec_integrals on primitives."""
    with pytest.raises(ValueError):
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.4]),
            np.array([[1.0]]),
        )

    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            1,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                two_int_brute(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
            ]
        ).reshape(3, 1, 1, 1, 1, 1, 1, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                two_int_brute(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, output=True),
            ]
        ).reshape(1, 3, 1, 1, 1, 1, 1, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                two_int_brute(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, output=True),
            ]
        ).reshape(1, 1, 3, 1, 1, 1, 1, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            0,
            np.array([[0, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            1,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, output=True),
                two_int_brute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, output=True),
            ]
        ).reshape(1, 1, 1, 3, 1, 1, 1, 1),
    )

    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            2,
            np.array([[0, 0, 2]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            2,
            np.array([[2, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        two_int_brute(0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1, 1, 0, output=True),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[2.0]]),
            np.array([1.0, 1.5, 2.0]),
            2,
            np.array([[0, 0, 2]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            2,
            np.array([[2, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        2 * two_int_brute(0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1, 1, 0, output=True),
    )


def test_compute_two_elec_integrals_segmented_contractions():
    """Test gbasis.integrals._two_elec_int._compute_two_elec_integrals on segmented contractions."""
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1, 0.15]),
            np.array([[1.0], [0.5]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.1)
        + 0.5 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.15),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2, 0.25]),
            np.array([[1.0], [0.6]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.2)
        + 0.6 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.25),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3, 0.35]),
            np.array([[1.0], [0.7]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.3)
        + 0.7 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.35),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4, 0.45]),
            np.array([[1.0], [0.8]]),
        ),
        1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.4)
        + 0.8 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.45),
    )


def test_compute_two_elec_integrals_generalized_contractions():
    """Test integrals._two_elec_int._compute_two_elec_integrals on generalized contractions."""
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1, 0.15]),
            np.array([[1.0, 2.0], [0.5, 1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.1)
                + 0.5
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.15),
                2 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.1)
                + 1.0
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, alpha=0.15),
            ]
        ).reshape(1, 1, 1, 1, 2, 1, 1, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2, 0.25]),
            np.array([[1.0, 0.9], [0.6, 0.5]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.2)
                + 0.6
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.25),
                0.9 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.2)
                + 0.5
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, beta=0.25),
            ]
        ).reshape(1, 1, 1, 1, 1, 2, 1, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3, 0.35]),
            np.array([[1.0, 0.8], [0.7, 0.8]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4]),
            np.array([[1.0]]),
        ),
        np.array(
            [
                1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.3)
                + 0.7
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.35),
                0.8 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.3)
                + 0.8
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, gamma=0.35),
            ]
        ).reshape(1, 1, 1, 1, 1, 1, 2, 1),
    )
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.2, 0.4, 0.6]),
            2,
            np.array([[0, 1, 1]]),
            np.array([0.1]),
            np.array([[1.0]]),
            np.array([1.0, 1.5, 2.0]),
            1,
            np.array([[0, 0, 1]]),
            np.array([0.2]),
            np.array([[1.0]]),
            np.array([0.1, 0.3, 0.5]),
            1,
            np.array([[1, 0, 0]]),
            np.array([0.3]),
            np.array([[1.0]]),
            np.array([1.1, 1.6, 2.1]),
            3,
            np.array([[1, 1, 1]]),
            np.array([0.4, 0.45]),
            np.array([[1.0, 1.1], [0.8, 0.9]]),
        ),
        np.array(
            [
                1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.4)
                + 0.8
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.45),
                1.1 * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.4)
                + 0.9
                * two_int_brute(0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, output=True, delta=0.45),
            ]
        ),
    )


@pytest.mark.xfail(
    reason="Known Bug: integrals with larger angular momentum and exponents results in an overflow."
)
def test_compute_two_elec_integrals_overflow_horton():
    """Test compute_two_elec_integrals against horton results for large exponents.

    Note that the output of _compute_two_elec_integrals does not normalize the contractions.

    """
    assert np.allclose(
        _compute_two_elec_integrals(
            boys_func,
            np.array([0.0, 0.0, 0.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array(
                [
                    1.8861445e02,
                    2.8276596e01,
                    6.4248300e00,
                    1.8150410e00,
                    5.9106300e-01,
                    2.1214900e-01,
                    7.9891000e-02,
                    2.7962000e-02,
                ]
            ),
            np.array(
                [
                    [-0.0212892],
                    [-0.1095596],
                    [-1.481826],
                    [3.0272963],
                    [-3.763086],
                    [3.6574131],
                    [-2.501237],
                    [0.89405394],
                ]
            ),
            np.array([0.8, 0.0, 0.0]),
            0,
            np.array([[0, 0, 0]]),
            np.array(
                [
                    1.1492705e03,
                    1.7204457e02,
                    3.9080465e01,
                    1.1036173e01,
                    3.5641520e00,
                    1.2404430e00,
                    4.4731600e-01,
                    1.6420600e-01,
                    5.7472000e-02,
                ]
            ),
            np.array(
                [
                    [0.01263588],
                    [-0.0848571],
                    [1.9186237],
                    [-2.949231],
                    [2.8122735],
                    [-2.270835],
                    [1.6543987],
                    [-1.000829],
                    [0.36960362],
                ]
            ),
            np.array([0.8, 0.0, 0.0]),
            3,
            np.array([[3, 0, 0]]),
            np.array([0.725333]),
            np.array([[1.0]]),
            np.array([0.8, 0.0, 0.0]),
            3,
            np.array([[3, 0, 0]]),
            np.array([0.725333]),
            np.array([[1.0]]),
        ),
        -0.02080288848109975,
    )
