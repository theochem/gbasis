"""Test gbasis.lincomb."""
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.lincomb import lincomb_blocks_evals
import numpy as np
import pytest


# TODO: add actual test case with real evaluation function
def test_lincomb_blocks_evals():
    """Test lincomb.lincomb_blocks_evals."""
    test = ContractedCartesianGaussians(
        2, np.array([0, 0, 0]), 0, np.array([0.0, 1.0]), np.array([0.1, 0.04])
    )

    def func(contraction):
        """Temporary function."""
        return np.ones((6, 7, 8))

    trans_blocks = [np.identity(6), np.identity(6)]

    with pytest.raises(TypeError):
        lincomb_blocks_evals({test, test}, func, trans_blocks)
    with pytest.raises(TypeError):
        lincomb_blocks_evals((test for i in range(2)), func, trans_blocks)

    with pytest.raises(TypeError):
        lincomb_blocks_evals([test], func, 0)
    with pytest.raises(TypeError):
        lincomb_blocks_evals([test], func, "123")

    with pytest.raises(TypeError):
        lincomb_blocks_evals([test, 1], func, trans_blocks)

    with pytest.raises(TypeError):
        lincomb_blocks_evals([test, test, test], func, trans_blocks + [np.arange(12, 18)])
    with pytest.raises(TypeError):
        lincomb_blocks_evals(
            [test, test, test], func, trans_blocks + [np.arange(12, 18).reshape(1, 6).tolist()]
        )

    with pytest.raises(ValueError):
        lincomb_blocks_evals(
            [test, test, test], func, trans_blocks + [np.arange(12, 19).reshape(1, 7)]
        )

    with pytest.raises(ValueError):
        lincomb_blocks_evals([test, test], func, trans_blocks + [np.arange(12, 18).reshape(1, 6)])

    with pytest.raises(ValueError):
        lincomb_blocks_evals([test, test, test], func, trans_blocks)

    def bad_func(contraction):
        """Temporary function that will raise a ValueError in lincomb_blocks_evals."""
        return np.random.rand(contraction.num_contr, contraction.num_contr, 8)

    test2 = ContractedCartesianGaussians(
        3, np.array([0, 0, 0]), 0, np.array([0.0, 1.0]), np.array([0.1, 0.04])
    )
    with pytest.raises(ValueError):
        lincomb_blocks_evals([test, test, test2], bad_func, trans_blocks + [np.identity(10)])

    assert np.allclose(lincomb_blocks_evals([test, test], func, trans_blocks), np.ones((12, 7, 8)))
