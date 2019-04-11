"""Test gbasis.density."""
from gbasis.contractions import make_contractions
from gbasis.density import eval_density_using_basis, eval_density_using_evaluated_orbs
from gbasis.eval import evaluate_basis_spherical_lincomb
from gbasis.parsers import parse_nwchem
import numpy as np
import pytest
from utils import find_datafile


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


def test_eval_density_using_basis():
    """Test gbasis.density.eval_density_using_basis."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    coords = np.random.rand(10, 3)

    eval_orbs = evaluate_basis_spherical_lincomb(basis, coords, transform)
    assert np.allclose(
        eval_density_using_basis(density, basis, coords, transform),
        np.einsum("ij,ik,jk->k", density, eval_orbs, eval_orbs),
    )
