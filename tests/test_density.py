"""Test gbasis.density."""
from gbasis.density import eval_density
import numpy as np
import pytest


def test_eval_density():
    """Test gbasis.density.eval_density."""
    density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    orb_eval = np.array([[1.0], [2.0]])
    assert np.allclose(eval_density(density_mat, orb_eval), np.array([27.0]))
    density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    orb_eval = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(eval_density(density_mat, orb_eval), np.array([85.0, 154.0, 243.0]))
    density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    orb_eval = np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]])
    assert np.allclose(
        eval_density(density_mat, orb_eval),
        np.array([[232.0, 468.0, 784.0], [340.0, 616.0, 972.0]]),
    )

    with pytest.raises(TypeError):
        orb_eval = [[1.0, 2.0], [1.0, 2.0]]
        eval_density(density_mat, orb_eval)
    with pytest.raises(TypeError):
        orb_eval = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density(density_mat, orb_eval)

    orb_eval = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(TypeError):
        density_mat = [[1.0, 2.0], [1.0, 2.0]]
        eval_density(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([1.0, 2.0, 3.0])
        eval_density(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        eval_density(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        eval_density(density_mat, orb_eval)
