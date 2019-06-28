"""Test gbasis.evals.evaluate_deriv."""
import itertools as it

from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.evals.eval_deriv import EvalDeriv, evaluate_deriv_basis
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_evaluate_deriv_construct_array_contraction():
    """Test gbasis.evals.evaluate_deriv.EvalDeriv.construct_array_contraction."""
    points = np.array([[2, 3, 4]])
    orders = np.array([0, 0, 0])
    contractions = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            points=[[2, 3, 4]], orders=orders, contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            points=points, orders=[0, 0, 0], contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            points=points, orders=orders, contractions=contractions.__dict__
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            points=points.reshape(3, 1), orders=orders, contractions=contractions
        )
    with pytest.raises(TypeError):
        EvalDeriv.construct_array_contraction(
            points=points, orders=orders.reshape(1, 3), contractions=contractions
        )
    with pytest.raises(ValueError):
        EvalDeriv.construct_array_contraction(
            points=points, orders=np.array([-1, 0, 0]), contractions=contractions
        )
    with pytest.raises(ValueError):
        EvalDeriv.construct_array_contraction(
            points=points, orders=np.array([0.0, 0, 0]), contractions=contractions
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
                for angmom_comp in test.angmom_components_cart
            ]
        ).reshape(3, 1)
        assert np.allclose(
            EvalDeriv.construct_array_contraction(
                points=np.array([[2, 3, 4]]), orders=orders, contractions=test
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
                for angmom_comp in test.angmom_components_cart
            ]
        ).reshape(3, 1)
        assert np.allclose(
            EvalDeriv.construct_array_contraction(
                points=np.array([[2, 3, 4]]), orders=orders, contractions=test
            ),
            answer,
        )


def test_evaluate_deriv_basis_cartesian():
    """Test gbasis.evals.basis_eval.evaluate_deriv_basis_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = EvalDeriv(basis)
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(
            points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), orders=np.array([0, 0, 0]), coord_type="cartesian"
        ),
    )
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(
            points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), orders=np.array([2, 1, 0]), coord_type="cartesian"
        ),
    )


def test_evaluate_deriv_basis_spherical():
    """Test gbasis.evals.basis_eval.evaluate_deriv_basis_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = EvalDeriv(basis)
    assert np.allclose(
        evaluate_obj.construct_array_spherical(
            points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), coord_type="spherical"
        ),
    )
    assert np.allclose(
        evaluate_obj.construct_array_spherical(
            points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), coord_type="spherical"
        ),
    )


def test_evaluate_deriv_basis_mix():
    """Test gbasis.evals.basis_eval.evaluate_deriv_basis_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = EvalDeriv(basis)
    assert np.allclose(
        evaluate_obj.construct_array_mix(
            ["cartesian"] * 8, points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), coord_type=["cartesian"] * 8
        ),
    )
    assert np.allclose(
        evaluate_obj.construct_array_mix(
            ["spherical"] * 8, points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), coord_type=["spherical"] * 8
        ),
    )


def test_evaluate_deriv_basis_lincomb():
    """Test gbasis.evals.basis_eval.evaluate_deriv_basis_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = EvalDeriv(basis)
    cart_transform = np.random.rand(14, 19)
    sph_transform = np.random.rand(14, 18)
    assert np.allclose(
        evaluate_obj.construct_array_lincomb(
            cart_transform, "cartesian", points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(
            basis,
            np.array([[1, 1, 1]]),
            np.array([0, 0, 0]),
            cart_transform,
            coord_type="cartesian",
        ),
    )
    assert np.allclose(
        evaluate_obj.construct_array_lincomb(
            sph_transform, "spherical", points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(
            basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), sph_transform, coord_type="spherical"
        ),
    )
