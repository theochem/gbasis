"""Test gbasis.eval."""
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.deriv import _eval_deriv_contractions
from gbasis.eval import Eval
import numpy as np
import pytest
from scipy.special import factorial2


def test_eval_shell():
    """Test gbasis.deriv.eval_shell."""
    test = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    answer = np.array(
        [
            _eval_deriv_contractions(
                np.array([[2, 3, 4]]),
                np.array([0, 0, 0]),
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
        Eval.construct_array_contraction(coords=np.array([[2, 3, 4]]), contractions=test), answer
    )

    with pytest.raises(TypeError):
        Eval.construct_array_contraction(coords=np.array([[2, 3, 4]]), contractions=None)
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(coords=np.array([[2, 3, 4]]), contractions={1: 2})
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(coords=np.array([2, 3, 4]), contractions=test)
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(coords=np.array([[3, 4]]), contractions=test)
