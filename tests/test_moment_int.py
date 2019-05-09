"""Test gbasis._moment_int."""
import itertools as it

from gbasis._moment_int import _compute_multipole_moment_integrals
import numpy as np
from scipy.special import factorial2


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
    x_pa = (0.1 * 0.2 + 0.2 * 1) / 0.3 - 0.2
    y_pa = (0.1 * 0.4 + 0.2 * 1.5) / 0.3 - 0.4
    z_pa = (0.1 * 0.6 + 0.2 * 2) / 0.3 - 0.6
    x_pb = (0.1 * 0.2 + 0.2 * 1) / 0.3 - 1.0
    y_pb = (0.1 * 0.4 + 0.2 * 1.5) / 0.3 - 1.5
    z_pb = (0.1 * 0.6 + 0.2 * 2) / 0.3 - 2.0
    x_pc = (0.1 * 0.2 + 0.2 * 1) / 0.3 - 0.0
    y_pc = (0.1 * 0.4 + 0.2 * 1.5) / 0.3 - 1.0
    z_pc = (0.1 * 0.6 + 0.2 * 2) / 0.3 - 2.0

    output = {}
    output[("x", 0, 0, 0)] = np.sqrt(np.pi / 0.3) * np.exp(-0.02 / 0.3 * 0.8 ** 2)
    output[("y", 0, 0, 0)] = np.sqrt(np.pi / 0.3) * np.exp(-0.02 / 0.3 * 1.1 ** 2)
    output[("z", 0, 0, 0)] = np.sqrt(np.pi / 0.3) * np.exp(-0.02 / 0.3 * 1.4 ** 2)
    output[("x", 0, 1, 0)] = x_pb * output[("x", 0, 0, 0)]
    output[("y", 0, 1, 0)] = y_pb * output[("y", 0, 0, 0)]
    output[("z", 0, 1, 0)] = z_pb * output[("z", 0, 0, 0)]
    output[("x", 0, 2, 0)] = x_pb * output[("x", 0, 1, 0)] + 1 * output["x", 0, 0, 0] / 2 / 0.3
    output[("y", 0, 2, 0)] = y_pb * output[("y", 0, 1, 0)] + 1 * output["y", 0, 0, 0] / 2 / 0.3
    output[("z", 0, 2, 0)] = z_pb * output[("z", 0, 1, 0)] + 1 * output["z", 0, 0, 0] / 2 / 0.3
    output[("x", 0, 3, 0)] = x_pb * output[("x", 0, 2, 0)] + 2 * output["x", 0, 1, 0] / 2 / 0.3
    output[("y", 0, 3, 0)] = y_pb * output[("y", 0, 2, 0)] + 2 * output["y", 0, 1, 0] / 2 / 0.3
    output[("z", 0, 3, 0)] = z_pb * output[("z", 0, 2, 0)] + 2 * output["z", 0, 1, 0] / 2 / 0.3

    output[("x", 1, 0, 0)] = x_pa * output[("x", 0, 0, 0)]
    output[("y", 1, 0, 0)] = y_pa * output[("y", 0, 0, 0)]
    output[("z", 1, 0, 0)] = z_pa * output[("z", 0, 0, 0)]
    output[("x", 1, 1, 0)] = x_pb * output[("x", 1, 0, 0)] + 1 * output["x", 0, 0, 0] / 2 / 0.3
    output[("y", 1, 1, 0)] = y_pb * output[("y", 1, 0, 0)] + 1 * output["y", 0, 0, 0] / 2 / 0.3
    output[("z", 1, 1, 0)] = z_pb * output[("z", 1, 0, 0)] + 1 * output["z", 0, 0, 0] / 2 / 0.3
    output[("x", 1, 2, 0)] = (
        x_pb * output[("x", 1, 1, 0)]
        + (1 * output["x", 0, 1, 0] + 1 * output["x", 1, 0, 0]) / 2 / 0.3
    )
    output[("y", 1, 2, 0)] = (
        y_pb * output[("y", 1, 1, 0)]
        + (1 * output["y", 0, 1, 0] + 1 * output["y", 1, 0, 0]) / 2 / 0.3
    )
    output[("z", 1, 2, 0)] = (
        z_pb * output[("z", 1, 1, 0)]
        + (1 * output["z", 0, 1, 0] + 1 * output["z", 1, 0, 0]) / 2 / 0.3
    )
    output[("x", 1, 3, 0)] = (
        x_pb * output[("x", 1, 2, 0)]
        + (1 * output["x", 0, 2, 0] + 2 * output["x", 1, 1, 0]) / 2 / 0.3
    )
    output[("y", 1, 3, 0)] = (
        y_pb * output[("y", 1, 2, 0)]
        + (1 * output["y", 0, 2, 0] + 2 * output["y", 1, 1, 0]) / 2 / 0.3
    )
    output[("z", 1, 3, 0)] = (
        z_pb * output[("z", 1, 2, 0)]
        + (1 * output["z", 0, 2, 0] + 2 * output["z", 1, 1, 0]) / 2 / 0.3
    )

    output[("x", 1, 3, 1)] = (
        x_pc * output[("x", 1, 3, 0)]
        + (1 * output["x", 0, 3, 0] + 3 * output["x", 1, 2, 0]) / 2 / 0.3
    )
    output[("y", 1, 3, 1)] = (
        y_pc * output[("y", 1, 3, 0)]
        + (1 * output["y", 0, 3, 0] + 3 * output["y", 1, 2, 0]) / 2 / 0.3
    )
    output[("z", 1, 3, 1)] = (
        z_pc * output[("z", 1, 3, 0)]
        + (1 * output["z", 0, 3, 0] + 3 * output["z", 1, 2, 0]) / 2 / 0.3
    )

    output[("x", 2, 0, 0)] = x_pa * output[("x", 1, 0, 0)] + 1 * output["x", 0, 0, 0] / 2 / 0.3
    output[("y", 2, 0, 0)] = y_pa * output[("y", 1, 0, 0)] + 1 * output["y", 0, 0, 0] / 2 / 0.3
    output[("z", 2, 0, 0)] = z_pa * output[("z", 1, 0, 0)] + 1 * output["z", 0, 0, 0] / 2 / 0.3
    output[("x", 2, 1, 0)] = x_pb * output[("x", 2, 0, 0)] + 2 * output["x", 1, 0, 0] / 2 / 0.3
    output[("y", 2, 1, 0)] = y_pb * output[("y", 2, 0, 0)] + 2 * output["y", 1, 0, 0] / 2 / 0.3
    output[("z", 2, 1, 0)] = z_pb * output[("z", 2, 0, 0)] + 2 * output["z", 1, 0, 0] / 2 / 0.3
    output[("x", 2, 2, 0)] = (
        x_pb * output[("x", 2, 1, 0)]
        + (2 * output["x", 1, 1, 0] + 1 * output["x", 2, 0, 0]) / 2 / 0.3
    )
    output[("y", 2, 2, 0)] = (
        y_pb * output[("y", 2, 1, 0)]
        + (2 * output["y", 1, 1, 0] + 1 * output["y", 2, 0, 0]) / 2 / 0.3
    )
    output[("z", 2, 2, 0)] = (
        z_pb * output[("z", 2, 1, 0)]
        + (2 * output["z", 1, 1, 0] + 1 * output["z", 2, 0, 0]) / 2 / 0.3
    )
    output[("x", 2, 3, 0)] = (
        x_pb * output[("x", 2, 2, 0)]
        + (2 * output["x", 1, 2, 0] + 2 * output["x", 2, 1, 0]) / 2 / 0.3
    )
    output[("y", 2, 3, 0)] = (
        y_pb * output[("y", 2, 2, 0)]
        + (2 * output["y", 1, 2, 0] + 2 * output["y", 2, 1, 0]) / 2 / 0.3
    )
    output[("z", 2, 3, 0)] = (
        z_pb * output[("z", 2, 2, 0)]
        + (2 * output["z", 1, 2, 0] + 2 * output["z", 2, 1, 0]) / 2 / 0.3
    )

    output[("x", 2, 2, 1)] = (
        x_pc * output[("x", 2, 2, 0)]
        + (2 * output["x", 1, 2, 0] + 2 * output["x", 2, 1, 0]) / 2 / 0.3
    )
    output[("y", 2, 2, 1)] = (
        y_pc * output[("y", 2, 2, 0)]
        + (2 * output["y", 1, 2, 0] + 2 * output["y", 2, 1, 0]) / 2 / 0.3
    )
    output[("z", 2, 2, 1)] = (
        z_pc * output[("z", 2, 2, 0)]
        + (2 * output["z", 1, 2, 0] + 2 * output["z", 2, 1, 0]) / 2 / 0.3
    )
    output[("x", 2, 3, 1)] = (
        x_pc * output[("x", 2, 3, 0)]
        + (2 * output["x", 1, 3, 0] + 3 * output["x", 2, 2, 0]) / 2 / 0.3
    )
    output[("y", 2, 3, 1)] = (
        y_pc * output[("y", 2, 3, 0)]
        + (2 * output["y", 1, 3, 0] + 3 * output["y", 2, 2, 0]) / 2 / 0.3
    )
    output[("z", 2, 3, 1)] = (
        z_pc * output[("z", 2, 3, 0)]
        + (2 * output["z", 1, 3, 0] + 3 * output["z", 2, 2, 0]) / 2 / 0.3
    )
    output[("x", 2, 3, 2)] = (
        x_pc * output[("x", 2, 3, 1)]
        + (2 * output["x", 1, 3, 1] + 3 * output["x", 2, 2, 1] + 1 * output["x", 2, 3, 0]) / 2 / 0.3
    )
    output[("y", 2, 3, 2)] = (
        y_pc * output[("y", 2, 3, 1)]
        + (2 * output["y", 1, 3, 1] + 3 * output["y", 2, 2, 1] + 1 * output["y", 2, 3, 0]) / 2 / 0.3
    )
    output[("z", 2, 3, 2)] = (
        z_pc * output[("z", 2, 3, 1)]
        + (2 * output["z", 1, 3, 1] + 3 * output["z", 2, 2, 1] + 1 * output["z", 2, 3, 0]) / 2 / 0.3
    )

    output[("x", 3, 0, 0)] = x_pa * output[("x", 2, 0, 0)] + 2 * output["x", 1, 0, 0] / 2 / 0.3
    output[("y", 3, 0, 0)] = y_pa * output[("y", 2, 0, 0)] + 2 * output["y", 1, 0, 0] / 2 / 0.3
    output[("z", 3, 0, 0)] = z_pa * output[("z", 2, 0, 0)] + 2 * output["z", 1, 0, 0] / 2 / 0.3
    output[("x", 3, 1, 0)] = x_pb * output[("x", 3, 0, 0)] + 3 * output["x", 2, 0, 0] / 2 / 0.3
    output[("y", 3, 1, 0)] = y_pb * output[("y", 3, 0, 0)] + 3 * output["y", 2, 0, 0] / 2 / 0.3
    output[("z", 3, 1, 0)] = z_pb * output[("z", 3, 0, 0)] + 3 * output["z", 2, 0, 0] / 2 / 0.3
    output[("x", 3, 2, 0)] = (
        x_pb * output[("x", 3, 1, 0)]
        + (3 * output["x", 2, 1, 0] + 1 * output["x", 3, 0, 0]) / 2 / 0.3
    )
    output[("y", 3, 2, 0)] = (
        y_pb * output[("y", 3, 1, 0)]
        + (3 * output["y", 2, 1, 0] + 1 * output["y", 3, 0, 0]) / 2 / 0.3
    )
    output[("z", 3, 2, 0)] = (
        z_pb * output[("z", 3, 1, 0)]
        + (3 * output["z", 2, 1, 0] + 1 * output["z", 3, 0, 0]) / 2 / 0.3
    )
    output[("x", 3, 3, 0)] = (
        x_pb * output[("x", 3, 2, 0)]
        + (3 * output["x", 2, 2, 0] + 2 * output["x", 3, 1, 0]) / 2 / 0.3
    )
    output[("y", 3, 3, 0)] = (
        y_pb * output[("y", 3, 2, 0)]
        + (3 * output["y", 2, 2, 0] + 2 * output["y", 3, 1, 0]) / 2 / 0.3
    )
    output[("z", 3, 3, 0)] = (
        z_pb * output[("z", 3, 2, 0)]
        + (3 * output["z", 2, 2, 0] + 2 * output["z", 3, 1, 0]) / 2 / 0.3
    )

    output[("x", 3, 1, 1)] = (
        x_pc * output[("x", 3, 1, 0)]
        + (3 * output["x", 2, 1, 0] + 1 * output["x", 3, 0, 0]) / 2 / 0.3
    )
    output[("y", 3, 1, 1)] = (
        y_pc * output[("y", 3, 1, 0)]
        + (3 * output["y", 2, 1, 0] + 1 * output["y", 3, 0, 0]) / 2 / 0.3
    )
    output[("z", 3, 1, 1)] = (
        z_pc * output[("z", 3, 1, 0)]
        + (3 * output["z", 2, 1, 0] + 1 * output["z", 3, 0, 0]) / 2 / 0.3
    )
    output[("x", 3, 2, 1)] = (
        x_pc * output[("x", 3, 2, 0)]
        + (3 * output["x", 2, 2, 0] + 2 * output["x", 3, 1, 0]) / 2 / 0.3
    )
    output[("y", 3, 2, 1)] = (
        y_pc * output[("y", 3, 2, 0)]
        + (3 * output["y", 2, 2, 0] + 2 * output["y", 3, 1, 0]) / 2 / 0.3
    )
    output[("z", 3, 2, 1)] = (
        z_pc * output[("z", 3, 2, 0)]
        + (3 * output["z", 2, 2, 0] + 2 * output["z", 3, 1, 0]) / 2 / 0.3
    )
    output[("x", 3, 2, 2)] = (
        x_pc * output[("x", 3, 2, 1)]
        + (3 * output["x", 2, 2, 1] + 2 * output["x", 3, 1, 1] + 1 * output["x", 3, 2, 0]) / 2 / 0.3
    )
    output[("y", 3, 2, 2)] = (
        y_pc * output[("y", 3, 2, 1)]
        + (3 * output["y", 2, 2, 1] + 2 * output["y", 3, 1, 1] + 1 * output["y", 3, 2, 0]) / 2 / 0.3
    )
    output[("z", 3, 2, 2)] = (
        z_pc * output[("z", 3, 2, 1)]
        + (3 * output["z", 2, 2, 1] + 2 * output["z", 3, 1, 1] + 1 * output["z", 3, 2, 0]) / 2 / 0.3
    )
    output[("x", 3, 3, 1)] = (
        x_pc * output[("x", 3, 3, 0)]
        + (3 * output["x", 2, 3, 0] + 3 * output["x", 3, 2, 0]) / 2 / 0.3
    )
    output[("y", 3, 3, 1)] = (
        y_pc * output[("y", 3, 3, 0)]
        + (3 * output["y", 2, 3, 0] + 3 * output["y", 3, 2, 0]) / 2 / 0.3
    )
    output[("z", 3, 3, 1)] = (
        z_pc * output[("z", 3, 3, 0)]
        + (3 * output["z", 2, 3, 0] + 3 * output["z", 3, 2, 0]) / 2 / 0.3
    )
    output[("x", 3, 3, 2)] = (
        x_pc * output[("x", 3, 3, 1)]
        + (3 * output["x", 2, 3, 1] + 3 * output["x", 3, 2, 1] + 1 * output["x", 3, 3, 0]) / 2 / 0.3
    )
    output[("y", 3, 3, 2)] = (
        y_pc * output[("y", 3, 3, 1)]
        + (3 * output["y", 2, 3, 1] + 3 * output["y", 3, 2, 1] + 1 * output["y", 3, 3, 0]) / 2 / 0.3
    )
    output[("z", 3, 3, 2)] = (
        z_pc * output[("z", 3, 3, 1)]
        + (3 * output["z", 2, 3, 1] + 3 * output["z", 3, 2, 1] + 1 * output["z", 3, 3, 0]) / 2 / 0.3
    )
    output[("x", 3, 3, 3)] = (
        x_pc * output[("x", 3, 3, 2)]
        + (3 * output["x", 2, 3, 2] + 3 * output["x", 3, 2, 2] + 2 * output["x", 3, 3, 1]) / 2 / 0.3
    )
    output[("y", 3, 3, 3)] = (
        y_pc * output[("y", 3, 3, 2)]
        + (3 * output["y", 2, 3, 2] + 3 * output["y", 3, 2, 2] + 2 * output["y", 3, 3, 1]) / 2 / 0.3
    )
    output[("z", 3, 3, 3)] = (
        z_pc * output[("z", 3, 3, 2)]
        + (3 * output["z", 2, 3, 2] + 3 * output["z", 3, 2, 2] + 2 * output["z", 3, 3, 1]) / 2 / 0.3
    )

    return output[(coord_type, i, j, k)]


def test_compute_multipole_moment_integrals_prim_angmom_left_recursion():
    """Test recursion for the left primitive in moment_int._compute_multipole_moment_integrals."""
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([1, 1.5, 2])
    exps_a = np.array([0.1])
    exps_b = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    coeffs_b = np.array([[1.0]])
    angmoms_b = np.array([[0, 0, 0]])
    coord_moment = np.array([0.0, 1.0, 2.0])
    order_moment = np.array([[0, 0, 0]])
    norm_b = np.array([[(2 * 0.2 / np.pi) ** (3 / 4) * (4 * 0.2) ** (3 / 2)]])

    # zero moment, zero right angular momentum
    for ix, iy, iz in it.product(range(4), range(4), range(4)):
        norm_a = np.array(
            (2 * 0.1 / np.pi) ** (3 / 4)
            * ((4 * 0.1) ** (3 / 2))
            / np.sqrt(np.prod(factorial2(2 * np.array([ix, iy, iz]) - 1)))
        ).reshape(1, 1)
        assert np.allclose(
            _compute_multipole_moment_integrals(
                coord_moment,
                order_moment,
                coord_a,
                np.array([[ix, iy, iz]]),
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
            * answer_prim("x", ix, 0, 0)
            * answer_prim("y", iy, 0, 0)
            * answer_prim("z", iz, 0, 0),
        )


def test_compute_multipole_moment_integrals_prim_angmom_right_recursion():
    """Test recursion for the right primitive in moment_int._compute_multipole_moment_integrals."""
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([1, 1.5, 2])
    exps_a = np.array([0.1])
    exps_b = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    coeffs_b = np.array([[1.0]])
    angmoms_a = np.array([[3, 3, 3]])
    coord_moment = np.array([0.0, 1.0, 2.0])
    order_moment = np.array([[0, 0, 0]])
    norm_a = np.array(
        (2 * 0.1 / np.pi) ** (3 / 4)
        * ((4 * 0.1) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_a - 1)))
    ).reshape(1, 1)

    # zero moment, three left angular momentum
    for jx, jy, jz in it.product(range(4), range(4), range(4)):
        norm_b = np.array(
            (2 * 0.2 / np.pi) ** (3 / 4)
            * ((4 * 0.2) ** (3 / 2))
            / np.sqrt(np.prod(factorial2(2 * np.array([jx, jy, jz]) - 1)))
        ).reshape(1, 1)
        assert np.allclose(
            _compute_multipole_moment_integrals(
                coord_moment,
                order_moment,
                coord_a,
                angmoms_a,
                exps_a,
                coeffs_a,
                norm_a,
                coord_b,
                np.array([[jx, jy, jz]]),
                exps_b,
                coeffs_b,
                norm_b,
            ),
            norm_a
            * norm_b
            * answer_prim("x", 3, jx, 0)
            * answer_prim("y", 3, jy, 0)
            * answer_prim("z", 3, jz, 0),
        )


def test_compute_multipole_moment_integrals_prim_moment_recursion():
    """Test recursion for multipole moment in moment_int._compute_multipole_moment_integrals."""
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([1, 1.5, 2])
    exps_a = np.array([0.1])
    exps_b = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    coeffs_b = np.array([[1.0]])
    angmoms_a = np.array([[3, 3, 3]])
    angmoms_b = np.array([[3, 3, 3]])
    coord_moment = np.array([0.0, 1.0, 2.0])
    norm_a = np.array(
        (2 * 0.1 / np.pi) ** (3 / 4)
        * ((4 * 0.1) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_a - 1)))
    ).reshape(1, 1)
    norm_b = np.array(
        (2 * 0.2 / np.pi) ** (3 / 4)
        * ((4 * 0.2) ** (3 / 2))
        / np.sqrt(np.prod(factorial2(2 * angmoms_b - 1)))
    ).reshape(1, 1)

    # three left angular momentum, three right angular momentum
    for kx, ky, kz in it.product(range(4), range(4), range(4)):
        assert np.allclose(
            _compute_multipole_moment_integrals(
                coord_moment,
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
            * answer_prim("x", 3, 3, kx)
            * answer_prim("y", 3, 3, ky)
            * answer_prim("z", 3, 3, kz),
        )


def test_compute_multipole_moment_integrals_contraction_norm():
    """Test moment_int._compute_multipole_moment_integrals using the norm of contractions.

    Note
    ----
    The function itself `_compute_multipole_moment_integrals` is used to test the use case for
    contractions. It assumes that this function behaves correctly for primitives.

    """
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_moment = np.array([0.0, 1.0, 2.0])
    order_moment = np.array([[0, 0, 0]])

    angmoms_a = np.array([[0, 0, 0]])
    exps_a = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    norm_a = np.prod(
        np.sqrt(
            (2 * exps_a[np.newaxis, np.newaxis, :] / np.pi) ** (1 / 2)
            * (4 * exps_a[np.newaxis, np.newaxis, :]) ** angmoms_a[:, :, np.newaxis]
            / factorial2(2 * angmoms_a[:, :, np.newaxis] - 1)
        ),
        axis=1,
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0]]),
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0]]),
        ),
        norm_a ** -2,
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
        ),
        1,
    )

    angmoms_a = np.array([[3, 3, 3]])
    exps_a = np.array([0.2])
    coeffs_a = np.array([[1.0]])
    norm_a = np.prod(
        np.sqrt(
            (2 * exps_a[np.newaxis, np.newaxis, :] / np.pi) ** (1 / 2)
            * (4 * exps_a[np.newaxis, np.newaxis, :]) ** angmoms_a[:, :, np.newaxis]
            / factorial2(2 * angmoms_a[:, :, np.newaxis] - 1)
        ),
        axis=1,
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0]]),
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0]]),
        ),
        norm_a ** -2,
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
        ),
        1,
    )

    angmoms_a = np.array([[3, 3, 3]])
    exps_a = np.array([0.2, 0.4])
    coeffs_a = np.array([[1.0], [1.0]])
    norm_a = np.prod(
        np.sqrt(
            (2 * exps_a[np.newaxis, np.newaxis, :] / np.pi) ** (1 / 2)
            * (4 * exps_a[np.newaxis, np.newaxis, :]) ** angmoms_a[:, :, np.newaxis]
            / factorial2(2 * angmoms_a[:, :, np.newaxis] - 1)
        ),
        axis=1,
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            norm_a,
        ),
        1.0
        + 1.0
        + 2
        * norm_a[:, 0]
        * norm_a[:, 1]
        * _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a[0:1],
            coeffs_a[0:1],
            np.array([[1.0]]),
            coord_a,
            angmoms_a,
            exps_a[1:],
            coeffs_a[1:],
            np.array([[1.0]]),
        ),
    )
    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0, 1.0]]),
            coord_a,
            angmoms_a,
            exps_a,
            coeffs_a,
            np.array([[1.0, 1.0]]),
        ),
        norm_a[:, 0] ** (-2)
        + norm_a[:, 1] ** (-2)
        + 2
        * norm_a[:, 0] ** (-1)
        * norm_a[:, 1] ** (-1)
        * _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            exps_a[0:1],
            coeffs_a[0:1],
            norm_a[:, 0:1],
            coord_a,
            angmoms_a,
            exps_a[1:],
            coeffs_a[1:],
            norm_a[:, 1:],
        ),
    )


def test_compute_multipole_moment_integrals_contraction():
    """Test moment_int._compute_multipole_moment_integrals on contractions.

    Note
    ----
    The function itself `_compute_multipole_moment_integrals` is used to test the use case for
    contractions. It assumes that this function behaves correctly for primitives.

    """
    coord_a = np.array([0.2, 0.4, 0.6])
    coord_b = np.array([1, 1.5, 2])
    angmoms_a = np.array([[3, 3, 3]])
    angmoms_b = np.array([[3, 3, 3]])
    coord_moment = np.array([0.0, 1.0, 2.0])
    order_moment = np.array([[3, 3, 3]])

    exps_a = np.array([0.1, 0.01])
    exps_b = np.array([0.2, 0.02])
    coeffs_a = np.array([[1.0], [2.0]])
    coeffs_b = np.array([[3.0], [4.0]])
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

    assert np.allclose(
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
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
        _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            np.array([0.1]),
            np.array([[1.0]]),
            norm_a[:, 0:1],
            coord_b,
            angmoms_b,
            np.array([0.2]),
            np.array([[3.0]]),
            norm_b[:, 0:1],
        )
        + _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            np.array([0.01]),
            np.array([[2.0]]),
            norm_a[:, 1:],
            coord_b,
            angmoms_b,
            np.array([0.2]),
            np.array([[3.0]]),
            norm_b[:, 0:1],
        )
        + _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            np.array([0.1]),
            np.array([[1.0]]),
            norm_a[:, 1:],
            coord_b,
            angmoms_b,
            np.array([0.02]),
            np.array([[4.0]]),
            norm_b[:, 0:1],
        )
        + _compute_multipole_moment_integrals(
            coord_moment,
            order_moment,
            coord_a,
            angmoms_a,
            np.array([0.01]),
            np.array([[2.0]]),
            norm_a[:, 1:],
            coord_b,
            angmoms_b,
            np.array([0.02]),
            np.array([[4.0]]),
            norm_b[:, 1:],
        ),
    )


def test_compute_multipole_moment_integrals_multiarray():
    """Test _compute_multipole_moment_integrals for computing multiple cases simultaneously.

    Note
    ----
    The function itself `_compute_multipole_moment_integrals` is used to test the use case for
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
    coord_moment = np.array([0.25, 0.45, 0.65])
    orders_moment = np.array(
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
    test = _compute_multipole_moment_integrals(
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
    )
    assert test.shape == (orders_moment.shape[0], 1, angmoms_a.shape[0], 1, angmoms_b.shape[0])
    for i, order_moment in enumerate(orders_moment):
        for j, angmom_a in enumerate(angmoms_a):
            for k, angmom_b in enumerate(angmoms_b):
                _compute_multipole_moment_integrals(
                    coord_moment,
                    np.array([order_moment]),
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


def test_compute_multipole_moment_integrals_generalized_contraction():
    """Test moment_int._compute_multipole_moment_integrals for generalized contractions."""
    moment_orders = [[0, 0, 0], [1, 0, 0], [3, 1, 2]]
    angmoms = [[0, 0, 0], [0, 1, 0], [0, 2, 1], [3, 2, 1]]
    test = _compute_multipole_moment_integrals(
        np.array([2, 3, 4]),
        np.array(moment_orders),
        np.array([0.5, 1, 1.5]),
        np.array(angmoms),
        np.array([1, 2]),
        np.array([[3, 4], [5, 6]]),
        np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
        np.array([1.0, 1.5, 2.0]),
        np.array(angmoms),
        np.array([1.2, 2.2]),
        np.array([[3.2, 4.2], [5.2, 6.2]]),
        np.array([[1.1, 0.9], [1.1, 0.9], [1.1, 0.9], [1.1, 0.9]]),
    )
    answer = np.array(
        [
            [
                _compute_multipole_moment_integrals(
                    np.array([2, 3, 4]),
                    np.array(moment_orders),
                    np.array([0.5, 1, 1.5]),
                    np.array(angmoms),
                    np.array([1, 2]),
                    np.array([[3], [5]]),
                    np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
                    np.array([1.0, 1.5, 2.0]),
                    np.array(angmoms),
                    np.array([1.2, 2.2]),
                    np.array([[3.2], [5.2]]),
                    np.array([[1.1, 0.9], [1.1, 0.9], [1.1, 0.9], [1.1, 0.9]]),
                ),
                _compute_multipole_moment_integrals(
                    np.array([2, 3, 4]),
                    np.array(moment_orders),
                    np.array([0.5, 1, 1.5]),
                    np.array(angmoms),
                    np.array([1, 2]),
                    np.array([[3], [5]]),
                    np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
                    np.array([1.0, 1.5, 2.0]),
                    np.array(angmoms),
                    np.array([1.2, 2.2]),
                    np.array([[4.2], [6.2]]),
                    np.array([[1.1, 0.9], [1.1, 0.9], [1.1, 0.9], [1.1, 0.9]]),
                ),
            ],
            [
                _compute_multipole_moment_integrals(
                    np.array([2, 3, 4]),
                    np.array(moment_orders),
                    np.array([0.5, 1, 1.5]),
                    np.array(angmoms),
                    np.array([1, 2]),
                    np.array([[4], [6]]),
                    np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
                    np.array([1.0, 1.5, 2.0]),
                    np.array(angmoms),
                    np.array([1.2, 2.2]),
                    np.array([[3.2], [5.2]]),
                    np.array([[1.1, 0.9], [1.1, 0.9], [1.1, 0.9], [1.1, 0.9]]),
                ),
                _compute_multipole_moment_integrals(
                    np.array([2, 3, 4]),
                    np.array(moment_orders),
                    np.array([0.5, 1, 1.5]),
                    np.array(angmoms),
                    np.array([1, 2]),
                    np.array([[4], [6]]),
                    np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
                    np.array([1.0, 1.5, 2.0]),
                    np.array(angmoms),
                    np.array([1.2, 2.2]),
                    np.array([[4.2], [6.2]]),
                    np.array([[1.1, 0.9], [1.1, 0.9], [1.1, 0.9], [1.1, 0.9]]),
                ),
            ],
        ]
    )
    answer = np.squeeze(answer, (3, 5))
    answer = np.transpose(answer, (2, 0, 3, 1, 4))
    assert np.allclose(test, answer)
