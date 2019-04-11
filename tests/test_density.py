"""Test gbasis.density."""
from gbasis.density import eval_density_using_evaluated_orbs
import numpy as np
import pytest


def test_eval_density_using_evaluated_orbs():
    """Test gbasis.density.eval_density_using_evaluated_orbs."""
    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0], [2.0]])
    assert np.allclose(
        eval_density_using_evaluated_orbs(density_mat, orb_eval),
        np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval),
    )
    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(
        eval_density_using_evaluated_orbs(density_mat, orb_eval),
        np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval),
    )

    with pytest.raises(TypeError):
        orb_eval = [[1.0, 2.0], [1.0, 2.0]]
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        orb_eval = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density_using_evaluated_orbs(density_mat, orb_eval)

    orb_eval = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(TypeError):
        density_mat = [[1.0, 2.0], [1.0, 2.0]]
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([1.0, 2.0, 3.0])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
