"""Test gbasis.evals.evaluate_deriv."""
import itertools as it

from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.evals.eval_deriv import EvalDeriv, evaluate_deriv_basis
from gbasis.parsers import make_contractions, parse_nwchem, parse_gbs
from gbasis.utils import factorial2
import numpy as np
import pytest
from utils import find_datafile


def test_evaluate_deriv_construct_array_contraction():
    """Test gbasis.evals.evaluate_deriv.EvalDeriv.construct_array_contraction."""
    points = np.array([[2, 3, 4]])
    orders = np.array([0, 0, 0])
    contractions = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
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
            1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
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
            1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
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
    """Test gbasis.evals.eval.evaluate_deriv_basis_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    evaluate_obj = EvalDeriv(basis)
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(
            points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(basis, np.array([[1, 1, 1]]), orders=np.array([0, 0, 0]), screen_basis=False),
    )
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(
            points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(basis, np.array([[1, 1, 1]]), orders=np.array([2, 1, 0]), screen_basis=False),
    )


def test_evaluate_deriv_basis_spherical():
    """Test gbasis.evals.eval.evaluate_deriv_basis_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    evaluate_obj = EvalDeriv(basis)
    assert np.allclose(
        evaluate_obj.construct_array_spherical(
            points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), screen_basis=False),
    )
    assert np.allclose(
        evaluate_obj.construct_array_spherical(
            points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), screen_basis=False),
    )


def test_evaluate_deriv_basis_mix():
    """Test gbasis.evals.eval.evaluate_deriv_basis_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    cartesian_basis = make_contractions(
        basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["cartesian"] * 8
    )
    spherical_basis = make_contractions(
        basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8
    )
    evaluate_obj_cartesian = EvalDeriv(cartesian_basis)
    evaluate_obj_spherical = EvalDeriv(spherical_basis)
    assert np.allclose(
        evaluate_obj_cartesian.construct_array_mix(
            ["cartesian"] * 8, points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(cartesian_basis, np.array([[1, 1, 1]]), np.array([0, 0, 0]), screen_basis=False),
    )
    assert np.allclose(
        evaluate_obj_spherical.construct_array_mix(
            ["spherical"] * 8, points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(spherical_basis, np.array([[1, 1, 1]]), np.array([2, 1, 0])),
    )


def test_evaluate_deriv_basis_lincomb():
    """Test gbasis.evals.eval.evaluate_deriv_basis_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    cartesian_basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    spherical_basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    evaluate_obj_cartesian = EvalDeriv(cartesian_basis)
    evaluate_obj_spherical = EvalDeriv(spherical_basis)
    cart_transform = np.random.rand(14, 19)
    sph_transform = np.random.rand(14, 18)
    assert np.allclose(
        evaluate_obj_cartesian.construct_array_lincomb(
            cart_transform, ["cartesian"], points=np.array([[1, 1, 1]]), orders=np.array([0, 0, 0])
        ),
        evaluate_deriv_basis(
            cartesian_basis,
            np.array([[1, 1, 1]]),
            np.array([0, 0, 0]),
            cart_transform,
            screen_basis=False,
        ),
    )
    assert np.allclose(
        evaluate_obj_spherical.construct_array_lincomb(
            sph_transform, ["spherical"], points=np.array([[1, 1, 1]]), orders=np.array([2, 1, 0])
        ),
        evaluate_deriv_basis(
            spherical_basis, np.array([[1, 1, 1]]), np.array([2, 1, 0]), sph_transform, screen_basis=False
        ),
    )

@pytest.mark.parametrize("precision", [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
def test_evaluate_basis_deriv_screening_accuracy(precision):
    """Test basis set derivative evaluation screening."""
    
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    #  the screening tolerance needs to be 1e-4 times the desired precision
    tol_screen = precision * 1e-4
    basis_deriv_evaluation = evaluate_deriv_basis(contraction, grid_3d, orders=np.array([1, 1, 1]), tol_screen=tol_screen)
    basis_deriv_evaluation_no_screen = evaluate_deriv_basis(contraction, grid_3d, orders=np.array([1, 1, 1]), screen_basis=False)
    assert np.allclose(basis_deriv_evaluation, basis_deriv_evaluation_no_screen, atol=precision)