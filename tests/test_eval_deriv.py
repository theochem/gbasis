"""Test gbasis.eval_deriv."""
import itertools as it

from gbasis._deriv import _eval_deriv_contractions
from gbasis.contractions import GeneralizedContractionShell, make_contractions
from gbasis.eval_deriv import (
    EvalDeriv,
    evaluate_deriv_basis_cartesian,
    evaluate_deriv_basis_lincomb,
    evaluate_deriv_basis_mix,
    evaluate_deriv_basis_spherical,
)
from gbasis.parsers import parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_eval_deriv_construct_array_contraction():
    """Test gbasis.eval_deriv.EvalDeriv.construct_array_contraction."""
    coords = np.array([[2, 3, 4]])
    orders = np.array([0, 0, 0])
    contractions = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
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

        test = GeneralizedContractionShell(
            1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
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

        test = GeneralizedContractionShell(
            1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
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


def test_evaluate_deriv_basis_cartesian():
    """Test gbasis.eval.evaluate_deriv_basis_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = EvalDeriv(basis)
    assert np.allclose(
        eval_obj.construct_array_cartesian(
            coords=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis_cartesian(basis, np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])),
    )
    assert np.allclose(
        eval_obj.construct_array_cartesian(
            coords=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis_cartesian(basis, np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])),
    )


def test_evaluate_deriv_basis_spherical():
    """Test gbasis.eval.evaluate_deriv_basis_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = EvalDeriv(basis)
    assert np.allclose(
        eval_obj.construct_array_spherical(
            coords=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis_spherical(basis, np.array([[1, 1, 1]]), np.array([0, 0, 0])),
    )
    assert np.allclose(
        eval_obj.construct_array_spherical(
            coords=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis_spherical(basis, np.array([[1, 1, 1]]), np.array([2, 1, 0])),
    )


def test_evaluate_deriv_basis_mix():
    """Test gbasis.eval.evaluate_deriv_basis_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = EvalDeriv(basis)
    assert np.allclose(
        eval_obj.construct_array_mix(
            ["cartesian"] * 8, coords=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis_mix(
            basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), ["cartesian"] * 8
        ),
    )
    assert np.allclose(
        eval_obj.construct_array_mix(
            ["spherical"] * 8, coords=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis_mix(
            basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), ["spherical"] * 8
        ),
    )


def test_evaluate_deriv_basis_lincomb():
    """Test gbasis.eval.evaluate_deriv_basis_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    eval_obj = EvalDeriv(basis)
    cart_transform = np.random.rand(14, 19)
    sph_transform = np.random.rand(14, 18)
    assert np.allclose(
        eval_obj.construct_array_lincomb(
            cart_transform, "cartesian", coords=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis_lincomb(
            basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), cart_transform, "cartesian"
        ),
    )
    assert np.allclose(
        eval_obj.construct_array_lincomb(
            sph_transform, "spherical", coords=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis_lincomb(
            basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), sph_transform, "spherical"
        ),
    )
