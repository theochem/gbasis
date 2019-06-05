"""Test gbasis.eval."""
from gbasis._deriv import _eval_deriv_contractions
from gbasis.contractions import GeneralizedContractionShell
from gbasis.eval import Eval, evaluate_basis
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_eval_construct_array_contraction():
    """Test gbasis.eval.Eval.construct_array_contraction."""
    test = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
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
            for angmom_comp in test.angmom_components_cart
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


def test_evaluate_basis_cartesian():
    """Test gbasis.eval.evaluate_basis_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]))
    eval_obj = Eval(basis)
    assert np.allclose(
        eval_obj.construct_array_cartesian(coords=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="cartesian"),
    )


def test_evaluate_basis_spherical():
    """Test gbasis.eval.evaluate_basis_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    # cartesian and spherical are the same for s orbital
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]))
    eval_obj = Eval(basis)
    assert np.allclose(
        eval_obj.construct_array_cartesian(coords=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="spherical"),
    )
    # p orbitals are zero at center
    basis = make_contractions(basis_dict, ["Li"], np.array([[0, 0, 0]]))
    eval_obj = Eval(basis)
    assert np.allclose(
        eval_obj.construct_array_cartesian(coords=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="spherical"),
    )

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = Eval(basis)
    assert np.allclose(
        eval_obj.construct_array_spherical(coords=np.array([[1, 1, 1]])),
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type="spherical"),
    )


def test_evaluate_basis_mix():
    """Test gbasis.eval.evaluate_basis_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    # cartesian and spherical are the same for s orbital
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]))
    assert np.allclose(
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="spherical"),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type=["spherical"]),
    )
    assert np.allclose(
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="cartesian"),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type=["cartesian"]),
    )

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    assert np.allclose(
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type="spherical"),
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type=["spherical"] * 8),
    )
    assert np.allclose(
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type="cartesian"),
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type=["cartesian"] * 8),
    )


def test_evaluate_basis_lincomb():
    """Test gbasis.eval.evaluate_basis_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = Eval(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        eval_obj.construct_array_lincomb(transform, "spherical", coords=np.array([[1, 1, 1]])),
        evaluate_basis(basis, np.array([[1, 1, 1]]), transform=transform, coord_type="spherical"),
    )
