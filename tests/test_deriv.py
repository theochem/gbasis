"""Test gbasis.evals._deriv."""
import itertools as it

from gbasis.evals._deriv import _eval_deriv_contractions
import numpy as np
from utils import partial_deriv_finite_diff


def evaluate_deriv_prim(coord, orders, center, angmom_comps, alpha):
    """Return the evaluation of the derivative of a Gaussian primitive.

    Parameters
    ----------
    coord : np.ndarray(3,)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alpha : float
        Value of the exponential in the Gaussian primitive.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    Note
    ----
    If you want to evaluate the derivative of the contractions of the primitive, then use
    `gbasis.evals._deriv.evaluate_deriv_contraction` instead.

    """
    return _eval_deriv_contractions(
        coord.reshape(1, 3),
        orders,
        center,
        angmom_comps.reshape(1, 3),
        np.array([alpha]),
        np.array([1.0]),
        np.array([[1]]),
    )[0]


def evaluate_prim(coord, center, angmom_comps, alpha):
    """Return the evaluation of a Gaussian primitive.

    Parameters
    ----------
    coord : np.ndarray(3,)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alpha : float
        Value of the exponential in the Guassian primitive.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    Note
    ----
    If you want to evaluate the contractions of the primitive, then use
    `gbasis.evals._deriv.evaluate_contraction` instead.

    """
    return evaluate_deriv_prim(coord, np.zeros(angmom_comps.shape), center, angmom_comps, alpha)


def test_evaluate_prim():
    """Test gbasis.evals._deriv.evaluate_prim.

    Note that this also tests the no derivative case of gbasis.evals._deriv.evaluate_deriv_prim.
    """
    # angular momentum: 0
    assert np.allclose(
        evaluate_prim(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1), 1
    )
    assert np.allclose(
        evaluate_prim(np.array([1.0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1),
        np.exp(-1),
    )
    assert np.allclose(
        evaluate_prim(np.array([1.0, 2.0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1),
        np.exp(-1) * np.exp(-4),
    )
    assert np.allclose(
        evaluate_prim(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 1),
        np.exp(-1) * np.exp(-4) * np.exp(-9),
    )
    # angular momentum 1
    assert np.allclose(
        evaluate_prim(np.array([2.0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]), 1),
        2 * np.exp(-(2**2)),
    )
    # other angular momentum
    assert np.allclose(
        evaluate_prim(np.array([2.0, 0, 0]), np.array([0, 3, 4]), np.array([2, 1, 3]), 1),
        4 * 3 * 4**3 * np.exp(-(2**2 + 3**2 + 4**2)),
    )


def test_evaluate_deriv_prim():
    """Test gbasis.evals._deriv.evaluate_deriv_prim."""
    # first order
    for k in range(3):
        orders = np.zeros(3, dtype=int)
        orders[k] = 1
        for x, y, z in it.product(range(3), range(3), range(3)):
            assert np.allclose(
                evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                ),
                partial_deriv_finite_diff(
                    lambda xyz: evaluate_prim(xyz, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1),
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
                evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                ),
                partial_deriv_finite_diff(
                    lambda xyz: evaluate_prim(xyz, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1),
                    np.array([2, 3, 4]),
                    orders,
                    epsilon=1e-5,
                    num_points=2,
                ),
            )


def test_evaluate_contractions():
    """Test gbasis.evals._deriv._evaluate_deriv_contraction without derivatization."""
    # angular momentum: 0
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[0, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([[0, 0, 0]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        1,
    )
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[1.0, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([[0, 0, 0]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        np.exp(-1),
    )
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[1.0, 2.0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([[0, 0, 0]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        np.exp(-1) * np.exp(-4),
    )
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[1.0, 2.0, 3.0]]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([[0, 0, 0]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        np.exp(-1) * np.exp(-4) * np.exp(-9),
    )
    # angular momentum 1
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[2.0, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([[1, 0, 0]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        2 * np.exp(-(2**2)),
    )
    # other angular momentum
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[2.0, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 3, 4]),
            np.array([[2, 1, 3]]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        4 * 3 * 4**3 * np.exp(-(2**2 + 3**2 + 4**2)),
    )
    # contraction
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[2, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 3, 4]),
            np.array([[2, 1, 3]]),
            np.array([0.1, 0.001]),
            np.array([3, 4]),
            np.array([[1.0, 1.0]]),
        ),
        3 * (2**2 * (-3) ** 1 * (-4) ** 3 * np.exp(-0.1 * (2**2 + 3**2 + 4**2)))
        + 4 * (2**2 * (-3) ** 1 * (-4) ** 3 * np.exp(-0.001 * (2**2 + 3**2 + 4**2))),
    )
    # contraction + multiple angular momentums
    assert np.allclose(
        _eval_deriv_contractions(
            np.array([[2, 0, 0]]),
            np.array([0, 0, 0]),
            np.array([0, 3, 4]),
            np.array([[2, 1, 3], [1, 3, 4]]),
            np.array([0.1, 0.001]),
            np.array([3, 4]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ),
        [
            [
                3 * (2**2 * (-3) ** 1 * (-4) ** 3 * np.exp(-0.1 * (2**2 + 3**2 + 4**2)))
                + 4 * (2**2 * (-3) ** 1 * (-4) ** 3 * np.exp(-0.001 * (2**2 + 3**2 + 4**2)))
            ],
            [
                3 * (2**1 * (-3) ** 3 * (-4) ** 4 * np.exp(-0.1 * (2**2 + 3**2 + 4**2)))
                + 4 * (2**1 * (-3) ** 3 * (-4) ** 4 * np.exp(-0.001 * (2**2 + 3**2 + 4**2)))
            ],
        ],
    )


def test_eval_deriv_contractions():
    """Test gbasis.evals._deriv._eval_deriv_contractions."""
    # first order
    for k in range(3):
        orders = np.zeros(3, dtype=int)
        orders[k] = 1
        for x, y, z in it.product(range(3), range(3), range(3)):
            # only contraction
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z]]),
                    np.array([1, 2]),
                    np.array([3, 4]),
                    np.array([[1, 1]]),
                ),
                3
                * evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                )
                + 4
                * evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 2
                ),
            )
            # contraction + multiple angular momentums
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z], [x - 1, y + 2, z + 1]]),
                    np.array([1, 2]),
                    np.array([3, 4]),
                    np.array([[1, 1]]),
                ),
                [
                    3
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                    )
                    + 4
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 2
                    ),
                    3
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]),
                        orders,
                        np.array([0.5, 1, 1.5]),
                        np.array([x - 1, y + 2, z + 1]),
                        1,
                    )
                    + 4
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]),
                        orders,
                        np.array([0.5, 1, 1.5]),
                        np.array([x - 1, y + 2, z + 1]),
                        2,
                    ),
                ],
            )
    # second order
    for k, l in it.product(range(3), range(3)):
        orders = np.zeros(3, dtype=int)
        orders[k] += 1
        orders[l] += 1
        for x, y, z in it.product(range(4), range(4), range(4)):
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z]]),
                    np.array([1]),
                    np.array([1]),
                    np.array([[1]]),
                ),
                evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                ),
            )
            # only contraction
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z]]),
                    np.array([1, 2]),
                    np.array([3, 4]),
                    np.array([[1, 1]]),
                ),
                3
                * evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                )
                + 4
                * evaluate_deriv_prim(
                    np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 2
                ),
            )
            # contraction + multiple angular momentums
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z], [x - 1, y + 2, z + 1]]),
                    np.array([1, 2]),
                    np.array([3, 4]),
                    np.array([[1, 1]]),
                ),
                [
                    3
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 1
                    )
                    + 4
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]), orders, np.array([0.5, 1, 1.5]), np.array([x, y, z]), 2
                    ),
                    3
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]),
                        orders,
                        np.array([0.5, 1, 1.5]),
                        np.array([x - 1, y + 2, z + 1]),
                        1,
                    )
                    + 4
                    * evaluate_deriv_prim(
                        np.array([2, 3, 4]),
                        orders,
                        np.array([0.5, 1, 1.5]),
                        np.array([x - 1, y + 2, z + 1]),
                        2,
                    ),
                ],
            )


def test_evaluate_deriv_generalized_contraction():
    """Test gbasis.evals._deriv._eval_deriv_contractions for generalized contractions."""
    for k, l in it.product(range(3), range(3)):
        orders = np.zeros(3, dtype=int)
        orders[k] += 1
        orders[l] += 1
        for x, y, z in it.product(range(4), range(4), range(4)):
            # only contraction
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z]]),
                    np.array([1, 2]),
                    np.array([[3, 4, 5], [6, 7, 8]]),
                    np.array([[1, 1]]),
                ),
                np.array(
                    [
                        3
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            1,
                        )
                        + 6
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            2,
                        ),
                        4
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            1,
                        )
                        + 7
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            2,
                        ),
                        5
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            1,
                        )
                        + 8
                        * evaluate_deriv_prim(
                            np.array([2, 3, 4]),
                            orders,
                            np.array([0.5, 1, 1.5]),
                            np.array([x, y, z]),
                            2,
                        ),
                    ]
                ).reshape(3, 1, 1),
            )
            # contraction + multiple angular momentums
            assert np.allclose(
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([[x, y, z], [x - 1, y + 2, z + 1]]),
                    np.array([1, 2]),
                    np.array([[3, 4, 5], [6, 7, 8]]),
                    np.array([[1, 1]]),
                ),
                np.array(
                    [
                        [
                            3
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                1,
                            )
                            + 6
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                2,
                            ),
                            3
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                1,
                            )
                            + 6
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                2,
                            ),
                        ],
                        [
                            4
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                1,
                            )
                            + 7
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                2,
                            ),
                            4
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                1,
                            )
                            + 7
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                2,
                            ),
                        ],
                        [
                            5
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                1,
                            )
                            + 8
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x, y, z]),
                                2,
                            ),
                            5
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                1,
                            )
                            + 8
                            * evaluate_deriv_prim(
                                np.array([2, 3, 4]),
                                orders,
                                np.array([0.5, 1, 1.5]),
                                np.array([x - 1, y + 2, z + 1]),
                                2,
                            ),
                        ],
                    ]
                ),
            )
