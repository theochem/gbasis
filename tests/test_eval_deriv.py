"""Test gbasis.eval_deriv."""
import itertools as it

from gbasis.contractions import ContractedCartesianGaussians
from gbasis.deriv import _eval_deriv_contractions
from gbasis.eval_deriv import EvalDeriv
import numpy as np
import pytest
from scipy.special import factorial2


def test_eval_deriv_construct_array_contraction():
    """Test gbasis.eval_deriv.EvalDeriv."""
    coords = np.array([[2, 3, 4]])
    orders = np.array([0, 0, 0])
    contractions = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            coords=[[2, 3, 4]], orders=orders, contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            coords=coords, orders=[0, 0, 0], contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            coords=coords, orders=orders, contractions=contractions.__dict__
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            coords=coords.reshape(3, 1), orders=orders, contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            coords=coords, orders=orders.reshape(1, 3), contractions=contractions
        )
    with pytest.raises(ValueError):
        EvalDeriv.construct_array_contraction(
            coords=coords, orders=np.array([-1, 0, 0]), contractions=contractions
        )
    with pytest.raises(ValueError):
        EvalDeriv.construct_array_contraction(
            coords=coords, orders=np.array([0.0, 0, 0]), contractions=contractions
        )

    # first order
    for k in range(3):
        orders = np.zeros(3, dtype=int)
        orders[k] = 1

        test = ContractedCartesianGaussians(
            1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
        )
        answer = np.array(
            [
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp]),
                    np.array([0.1, 0.01]),
                    np.array([1, 2]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp in test.angmom_components
            ]
        ).reshape(3, 1)
        assert np.allclose(
            EvalDeriv.construct_array_contraction(
                coords=np.array([[2, 3, 4]]), orders=orders, contractions=test
            ),
            answer,
        )
    # second order
    for k, l in it.product(range(3), range(3)):
        orders = np.zeros(3, dtype=int)
        orders[k] += 1
        orders[l] += 1

        test = ContractedCartesianGaussians(
            1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
        )
        answer = np.array(
            [
                _eval_deriv_contractions(
                    np.array([[2, 3, 4]]),
                    orders,
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp]),
                    np.array([0.1, 0.01]),
                    np.array([1, 2]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp in test.angmom_components
            ]
        ).reshape(3, 1)
        assert np.allclose(
            EvalDeriv.construct_array_contraction(
                coords=np.array([[2, 3, 4]]), orders=orders, contractions=test
            ),
            answer,
        )
