"""Test gbasis.evals.eval."""
from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.evals.eval import Eval, evaluate_basis
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile, HortonContractions


def test_evaluate_construct_array_contraction():
    """Test gbasis.evals.eval.Eval.construct_array_contraction."""
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
        Eval.construct_array_contraction(points=np.array([[2, 3, 4]]), contractions=test), answer
    )

    with pytest.raises(TypeError):
        Eval.construct_array_contraction(points=np.array([[2, 3, 4]]), contractions=None)
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(points=np.array([[2, 3, 4]]), contractions={1: 2})
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(points=np.array([2, 3, 4]), contractions=test)
    with pytest.raises(TypeError):
        Eval.construct_array_contraction(points=np.array([[3, 4]]), contractions=test)


def test_evaluate_basis_cartesian():
    """Test gbasis.evals.eval.evaluate_basis_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]))
    evaluate_obj = Eval(basis)
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(points=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="cartesian"),
    )


def test_evaluate_basis_spherical():
    """Test gbasis.evals.eval.evaluate_basis_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    # cartesian and spherical are the same for s orbital
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]))
    evaluate_obj = Eval(basis)
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(points=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="spherical"),
    )
    # p orbitals are zero at center
    basis = make_contractions(basis_dict, ["Li"], np.array([[0, 0, 0]]))
    evaluate_obj = Eval(basis)
    assert np.allclose(
        evaluate_obj.construct_array_cartesian(points=np.array([[0, 0, 0]])),
        evaluate_basis(basis, np.array([[0, 0, 0]]), coord_type="spherical"),
    )

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = Eval(basis)
    assert np.allclose(
        evaluate_obj.construct_array_spherical(points=np.array([[1, 1, 1]])),
        evaluate_basis(basis, np.array([[1, 1, 1]]), coord_type="spherical"),
    )


def test_evaluate_basis_mix():
    """Test gbasis.evals.eval.evaluate_basis_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

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
    """Test gbasis.evals.eval.evaluate_basis_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    evaluate_obj = Eval(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        evaluate_obj.construct_array_lincomb(transform, "spherical", points=np.array([[1, 1, 1]])),
        evaluate_basis(basis, np.array([[1, 1, 1]]), transform=transform, coord_type="spherical"),
    )


def test_evaluate_basis_horton():
    """Test gbasis.evals.eval.evaluate_basis against horton results."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    points = np.array([[0, 0, 0], [0.8, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_eval_cart = np.load(find_datafile("data_horton_hhe_cart_eval.npy"))
    horton_eval_sph = np.load(find_datafile("data_horton_hhe_sph_eval.npy"))

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(evaluate_basis(basis, grid_3d, coord_type="cartesian"), horton_eval_cart.T)
    assert np.allclose(evaluate_basis(basis, grid_3d, coord_type="spherical"), horton_eval_sph.T)
