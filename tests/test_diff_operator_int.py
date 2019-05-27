"""Test gbasis._diff_operator_int."""
import itertools as it

from gbasis._diff_operator_int import _compute_differential_operator_integrals
import numpy as np
from scipy.special import factorial2
from test_moment_int import answer_prim as answer_prim_overlap


def answer_prim(coord_type, i, j, k):
    """Return the answer to the multipole moment integral tests.

    Data for primitive on the left:
    - Coordinate: [0.2, 0.4, 0.6]
    - Exponents: [0.1]
    - Coefficients: [1.0]

    Data for primitive on the right:
    - Coordinate: [1.0, 1.5, 2.0]
    - Exponents: [0.2]
    - Coefficients: [1.0]

    Overlap parts were copied over from test_moment_int.answer_prim.

    Parameters
    ----------
    coord_type : {'x', 'y', 'z'}
        Coordinate along which the multipole moment is integrated.
    i : int
        Angular momentum component for the given coordinate of the primitive on the left.
    j : int
        Angular momentum component for the given coordinate of the primitive on the right.
    k : int
        Order of the multipole moment for the given coordinate.

    Returns
    -------
    answer : float

    """
    output = {}
    for a, b in it.product(range(4), range(4)):
        output[("x", a, b, 0)] = answer_prim_overlap("x", a, b, 0)
        output[("y", a, b, 0)] = answer_prim_overlap("y", a, b, 0)
        output[("z", a, b, 0)] = answer_prim_overlap("z", a, b, 0)

    output[("x", 0, 0, 1)] = 2 * 0.1 * output[("x", 1, 0, 0)]
    output[("y", 0, 0, 1)] = 2 * 0.1 * output[("y", 1, 0, 0)]
    output[("z", 0, 0, 1)] = 2 * 0.1 * output[("z", 1, 0, 0)]
    output[("x", 0, 1, 1)] = 2 * 0.1 * output[("x", 1, 1, 0)]
    output[("y", 0, 1, 1)] = 2 * 0.1 * output[("y", 1, 1, 0)]
    output[("z", 0, 1, 1)] = 2 * 0.1 * output[("z", 1, 1, 0)]
    output[("x", 0, 2, 1)] = 2 * 0.1 * output[("x", 1, 2, 0)]
    output[("y", 0, 2, 1)] = 2 * 0.1 * output[("y", 1, 2, 0)]
    output[("z", 0, 2, 1)] = 2 * 0.1 * output[("z", 1, 2, 0)]
    output[("x", 0, 3, 1)] = 2 * 0.1 * output[("x", 1, 3, 0)]
    output[("y", 0, 3, 1)] = 2 * 0.1 * output[("y", 1, 3, 0)]
    output[("z", 0, 3, 1)] = 2 * 0.1 * output[("z", 1, 3, 0)]

    output[("x", 1, 0, 1)] = 2 * 0.1 * output[("x", 2, 0, 0)] - 1 * output[("x", 0, 0, 0)]
    output[("y", 1, 0, 1)] = 2 * 0.1 * output[("y", 2, 0, 0)] - 1 * output[("y", 0, 0, 0)]
    output[("z", 1, 0, 1)] = 2 * 0.1 * output[("z", 2, 0, 0)] - 1 * output[("z", 0, 0, 0)]
    output[("x", 1, 1, 1)] = 2 * 0.1 * output[("x", 2, 1, 0)] - 1 * output[("x", 0, 1, 0)]
    output[("y", 1, 1, 1)] = 2 * 0.1 * output[("y", 2, 1, 0)] - 1 * output[("y", 0, 1, 0)]
    output[("z", 1, 1, 1)] = 2 * 0.1 * output[("z", 2, 1, 0)] - 1 * output[("z", 0, 1, 0)]
    output[("x", 1, 2, 1)] = 2 * 0.1 * output[("x", 2, 2, 0)] - 1 * output[("x", 0, 2, 0)]
    output[("y", 1, 2, 1)] = 2 * 0.1 * output[("y", 2, 2, 0)] - 1 * output[("y", 0, 2, 0)]
    output[("z", 1, 2, 1)] = 2 * 0.1 * output[("z", 2, 2, 0)] - 1 * output[("z", 0, 2, 0)]
    output[("x", 1, 3, 1)] = 2 * 0.1 * output[("x", 2, 3, 0)] - 1 * output[("x", 0, 3, 0)]
    output[("y", 1, 3, 1)] = 2 * 0.1 * output[("y", 2, 3, 0)] - 1 * output[("y", 0, 3, 0)]
    output[("z", 1, 3, 1)] = 2 * 0.1 * output[("z", 2, 3, 0)] - 1 * output[("z", 0, 3, 0)]

    output[("x", 2, 0, 1)] = 2 * 0.1 * output[("x", 3, 0, 0)] - 2 * output[("x", 1, 0, 0)]
    output[("y", 2, 0, 1)] = 2 * 0.1 * output[("y", 3, 0, 0)] - 2 * output[("y", 1, 0, 0)]
    output[("z", 2, 0, 1)] = 2 * 0.1 * output[("z", 3, 0, 0)] - 2 * output[("z", 1, 0, 0)]
    output[("x", 2, 1, 1)] = 2 * 0.1 * output[("x", 3, 1, 0)] - 2 * output[("x", 1, 1, 0)]
    output[("y", 2, 1, 1)] = 2 * 0.1 * output[("y", 3, 1, 0)] - 2 * output[("y", 1, 1, 0)]
    output[("z", 2, 1, 1)] = 2 * 0.1 * output[("z", 3, 1, 0)] - 2 * output[("z", 1, 1, 0)]
    output[("x", 2, 2, 1)] = 2 * 0.1 * output[("x", 3, 2, 0)] - 2 * output[("x", 1, 2, 0)]
    output[("y", 2, 2, 1)] = 2 * 0.1 * output[("y", 3, 2, 0)] - 2 * output[("y", 1, 2, 0)]
    output[("z", 2, 2, 1)] = 2 * 0.1 * output[("z", 3, 2, 0)] - 2 * output[("z", 1, 2, 0)]
    output[("x", 2, 3, 1)] = 2 * 0.1 * output[("x", 3, 3, 0)] - 2 * output[("x", 1, 3, 0)]
    output[("y", 2, 3, 1)] = 2 * 0.1 * output[("y", 3, 3, 0)] - 2 * output[("y", 1, 3, 0)]
    output[("z", 2, 3, 1)] = 2 * 0.1 * output[("z", 3, 3, 0)] - 2 * output[("z", 1, 3, 0)]

    output[("x", 0, 0, 2)] = 2 * 0.1 * output[("x", 1, 0, 1)]
    output[("y", 0, 0, 2)] = 2 * 0.1 * output[("y", 1, 0, 1)]
    output[("z", 0, 0, 2)] = 2 * 0.1 * output[("z", 1, 0, 1)]
    output[("x", 0, 1, 2)] = 2 * 0.1 * output[("x", 1, 1, 1)]
    output[("y", 0, 1, 2)] = 2 * 0.1 * output[("y", 1, 1, 1)]
    output[("z", 0, 1, 2)] = 2 * 0.1 * output[("z", 1, 1, 1)]
    output[("x", 0, 2, 2)] = 2 * 0.1 * output[("x", 1, 2, 1)]
    output[("y", 0, 2, 2)] = 2 * 0.1 * output[("y", 1, 2, 1)]
    output[("z", 0, 2, 2)] = 2 * 0.1 * output[("z", 1, 2, 1)]
    output[("x", 0, 3, 2)] = 2 * 0.1 * output[("x", 1, 3, 1)]
    output[("y", 0, 3, 2)] = 2 * 0.1 * output[("y", 1, 3, 1)]
    output[("z", 0, 3, 2)] = 2 * 0.1 * output[("z", 1, 3, 1)]

    output[("x", 1, 0, 2)] = 2 * 0.1 * output[("x", 2, 0, 1)] - 1 * output[("x", 0, 0, 1)]
    output[("y", 1, 0, 2)] = 2 * 0.1 * output[("y", 2, 0, 1)] - 1 * output[("y", 0, 0, 1)]
    output[("z", 1, 0, 2)] = 2 * 0.1 * output[("z", 2, 0, 1)] - 1 * output[("z", 0, 0, 1)]
    output[("x", 1, 1, 2)] = 2 * 0.1 * output[("x", 2, 1, 1)] - 1 * output[("x", 0, 1, 1)]
    output[("y", 1, 1, 2)] = 2 * 0.1 * output[("y", 2, 1, 1)] - 1 * output[("y", 0, 1, 1)]
    output[("z", 1, 1, 2)] = 2 * 0.1 * output[("z", 2, 1, 1)] - 1 * output[("z", 0, 1, 1)]
    output[("x", 1, 2, 2)] = 2 * 0.1 * output[("x", 2, 2, 1)] - 1 * output[("x", 0, 2, 1)]
    output[("y", 1, 2, 2)] = 2 * 0.1 * output[("y", 2, 2, 1)] - 1 * output[("y", 0, 2, 1)]
    output[("z", 1, 2, 2)] = 2 * 0.1 * output[("z", 2, 2, 1)] - 1 * output[("z", 0, 2, 1)]
    output[("x", 1, 3, 2)] = 2 * 0.1 * output[("x", 2, 3, 1)] - 1 * output[("x", 0, 3, 1)]
    output[("y", 1, 3, 2)] = 2 * 0.1 * output[("y", 2, 3, 1)] - 1 * output[("y", 0, 3, 1)]
    output[("z", 1, 3, 2)] = 2 * 0.1 * output[("z", 2, 3, 1)] - 1 * output[("z", 0, 3, 1)]

    output[("x", 0, 0, 3)] = 2 * 0.1 * output[("x", 1, 0, 2)]
    output[("y", 0, 0, 3)] = 2 * 0.1 * output[("y", 1, 0, 2)]
    output[("z", 0, 0, 3)] = 2 * 0.1 * output[("z", 1, 0, 2)]
    output[("x", 0, 1, 3)] = 2 * 0.1 * output[("x", 1, 1, 2)]
    output[("y", 0, 1, 3)] = 2 * 0.1 * output[("y", 1, 1, 2)]
    output[("z", 0, 1, 3)] = 2 * 0.1 * output[("z", 1, 1, 2)]
    output[("x", 0, 2, 3)] = 2 * 0.1 * output[("x", 1, 2, 2)]
    output[("y", 0, 2, 3)] = 2 * 0.1 * output[("y", 1, 2, 2)]
    output[("z", 0, 2, 3)] = 2 * 0.1 * output[("z", 1, 2, 2)]
    output[("x", 0, 3, 3)] = 2 * 0.1 * output[("x", 1, 3, 2)]
    output[("y", 0, 3, 3)] = 2 * 0.1 * output[("y", 1, 3, 2)]
    output[("z", 0, 3, 3)] = 2 * 0.1 * output[("z", 1, 3, 2)]

    return output[(coord_type, i, j, k)]


def test_compute_differential_operator_integrals_diff_recursion():
    """Test recursion for order of derivative in _compute_differential_operator_integrals."""
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([1, 1.5, 2])
    exps_a = np.array([0.1])
    exps_b = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    coeffs_b = np.array([[1.0]])
    angmoms_b = np.array([[3, 3, 3]])
    norm_b = np.array(
        (2 * 0.2 / np.pi) ** (3 / 4)
        * ((4 * 0.2) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_b - 1)))
    ).reshape(1, 1)

    angmoms_a = np.array([[0, 0, 0]])
    norm_a = np.array(
        (2 * 0.1 / np.pi) ** (3 / 4)
        * ((4 * 0.1) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_a - 1)))
    ).reshape(1, 1)
    for kx, ky, kz in it.product(range(4), range(4), range(4)):
        if kx == 0 and ky == 0 and kz == 0:
            continue
        assert np.allclose(
            _compute_differential_operator_integrals(
                np.array([[kx, ky, kz]]),
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
            ),
            norm_a
            * norm_b
            * answer_prim("x", 0, 3, kx)
            * answer_prim("y", 0, 3, ky)
            * answer_prim("z", 0, 3, kz),
        )

    angmoms_a = np.array([[1, 1, 1]])
    norm_a = np.array(
        (2 * 0.1 / np.pi) ** (3 / 4)
        * ((4 * 0.1) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_a - 1)))
    ).reshape(1, 1)
    for kx, ky, kz in it.product(range(3), range(3), range(3)):
        if kx == 0 and ky == 0 and kz == 0:
            continue
        assert np.allclose(
            _compute_differential_operator_integrals(
                np.array([[kx, ky, kz]]),
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
            ),
            norm_a
            * norm_b
            * answer_prim("x", 1, 3, kx)
            * answer_prim("y", 1, 3, ky)
            * answer_prim("z", 1, 3, kz),
        )

    angmoms_a = np.array([[2, 2, 2]])
    norm_a = np.array(
        (2 * 0.1 / np.pi) ** (3 / 4)
        * ((4 * 0.1) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_a - 1)))
    ).reshape(1, 1)
    for kx, ky, kz in it.product(range(2), range(2), range(2)):
        if kx == 0 and ky == 0 and kz == 0:
            continue
        assert np.allclose(
            _compute_differential_operator_integrals(
                np.array([[kx, ky, kz]]),
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
            ),
            norm_a
            * norm_b
            * answer_prim("x", 2, 3, kx)
            * answer_prim("y", 2, 3, ky)
            * answer_prim("z", 2, 3, kz),
        )


def test_compute_differential_operator_integrals_multiarray():
    """Test _compute_differential_operator_integrals for computing multiple cases simultaneously.

    Note
    ----
    The function itself `_compute_differential._operator_integrals` is used to test the use case for
    contractions. It assumes that this function behaves correctly for contractions.

    """
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([0.3, 0.5, 0.7])
    angmoms_a = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [2, 1, 0],
            [2, 0, 1],
            [1, 2, 0],
            [0, 2, 1],
            [1, 0, 2],
            [0, 1, 2],
        ]
    )
    angmoms_b = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    exps_a = np.array([5.4471780, 0.8245470])
    exps_b = np.array([0.1831920])
    coeffs_a = np.array([[0.1562850], [0.9046910]])
    coeffs_b = np.array([[1.0]])
    orders_diff = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [2, 1, 0],
            [2, 0, 1],
            [1, 2, 0],
            [0, 2, 1],
            [1, 0, 2],
            [0, 1, 2],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]
    )

    norm_a = np.prod(
        np.sqrt(
            (2 * exps_a[np.newaxis, np.newaxis, :] / np.pi) ** (1 / 2)
            * (4 * exps_a[np.newaxis, np.newaxis, :]) ** angmoms_a[:, :, np.newaxis]
            / factorial2(2 * angmoms_a[:, :, np.newaxis] - 1)
        ),
        axis=1,
    )
    norm_b = np.prod(
        np.sqrt(
            (2 * exps_b[np.newaxis, np.newaxis, :] / np.pi) ** (1 / 2)
            * (4 * exps_b[np.newaxis, np.newaxis, :]) ** angmoms_b[:, :, np.newaxis]
            / factorial2(2 * angmoms_b[:, :, np.newaxis] - 1)
        ),
        axis=1,
    )
    test = _compute_differential_operator_integrals(
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
    )
    assert test.shape == (orders_diff.shape[0], 1, angmoms_a.shape[0], 1, angmoms_b.shape[0])
    for i, order_diff in enumerate(orders_diff):
        for j, angmom_a in enumerate(angmoms_a):
            for k, angmom_b in enumerate(angmoms_b):
                _compute_differential_operator_integrals(
                    np.array([order_diff]),
                    coord_a,
                    np.array([angmom_a]),
                    exps_a,
                    coeffs_a,
                    norm_a,
                    coord_b,
                    np.array([angmom_b]),
                    exps_b,
                    coeffs_b,
                    norm_b,
                ) == test[i, 0, j, 0, k]
