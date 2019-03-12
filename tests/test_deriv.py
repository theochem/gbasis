"""Test gbasis.deriv."""
import itertools as it

from gbasis.deriv import eval_deriv_prim, eval_prim
import numpy as np
from utils import partial_deriv_finite_diff


def test_eval_prim():
    """Test gbasis.deriv.eval_prim.

    Note that this also tests the no derivative case of gbasis.deriv.eval_deriv_prim.
    """
    # angular momentum: 0
    assert np.allclose(
        eval_prim(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1), 1
    )
    assert np.allclose(
        eval_prim(np.array([1.0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1), np.exp(-1)
    )
    assert np.allclose(
        eval_prim(np.array([1.0, 2.0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1),
        np.exp(-1) * np.exp(-4),
    )
    assert np.allclose(
        eval_prim(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1),
        np.exp(-1) * np.exp(-4) * np.exp(-9),
    )
    # angular momentum 1
    assert np.allclose(
        eval_prim(np.array([2.0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]), 1),
        2 * np.exp(-2 ** 2),
    )
    # other angular momentum
    assert np.allclose(
        eval_prim(np.array([2.0, 0, 0]), np.array([0, 3, 4]), np.array([2, 1, 3]), 1),
        4 * 3 * 4 ** 3 * np.exp(-(2 ** 2 + 3 ** 2 + 4 ** 2)),
    )


def test_eval_deriv_prim():
    """Test gbasis.deriv.eval_deriv_prim."""
    # first order
    for k in range(3):
        orders = np.zeros(3, dtype=int)
        orders[k] = 1
        for x, y, z in it.product(range(3), range(3), range(3)):
            assert np.allclose(
                eval_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                ),
                partial_deriv_finite_diff(
                    lambda xyz: eval_prim(xyz, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1),
                    np.array([2, 3, 4]),
                    orders,
                ),
            )
    # second order
    for k, l in it.product(range(3), range(3)):
        orders = np.zeros(3, dtype=int)
        orders[k] += 1
        orders[l] += 1
        for x, y, z in it.product(range(4), range(4), range(4)):
            assert np.allclose(
                eval_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                ),
                partial_deriv_finite_diff(
                    lambda xyz: eval_prim(xyz, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1),
                    np.array([2, 3, 4]),
                    orders,
                    epsilon=1e-5,
                    num_points=2,
                ),
            )
