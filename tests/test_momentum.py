"""Test gbasis.integrals.momentum."""
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import _compute_differential_operator_integrals
from gbasis.integrals.momentum import momentum_integral, MomentumIntegral
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
from gbasis.utils import factorial2
import numpy as np
import pytest
from utils import find_datafile


def test_momentum_construct_array_contraction():
    """Test gbasis.integrals.momentum.MomentumIntegral.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02]), "spherical"
    )
    test = MomentumIntegral.construct_array_contraction(
        test_one, test_two, screen_basis=False
    ).squeeze()
    answer = np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[1, 0, 0]]),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components_cart
            ]
            for angmom_comp_one in test_one.angmom_components_cart
        ]
    )
    assert np.allclose(test[:, :, 0], -1j * answer.squeeze())

    answer = np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[0, 1, 0]]),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components_cart
            ]
            for angmom_comp_one in test_one.angmom_components_cart
        ]
    )
    assert np.allclose(test[:, :, 1], -1j * answer.squeeze())

    answer = np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[0, 0, 1]]),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components_cart
            ]
            for angmom_comp_one in test_one.angmom_components_cart
        ]
    )
    assert np.allclose(test[:, :, 2], -1j * answer.squeeze())

    with pytest.raises(TypeError):
        MomentumIntegral.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        MomentumIntegral.construct_array_contraction(None, test_two)


def test_momentum_integral_cartesian():
    """Test gbasis.integrals.momentum.momentum_integral_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_cartesian(screen_basis=False),
        momentum_integral(basis, screen_basis=False),
    )


def test_momentum_integral_spherical():
    """Test gbasis.integrals.momentum.momentum_integral_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_spherical(screen_basis=False),
        momentum_integral(basis, screen_basis=False),
    )


def test_momentum_integral_mix():
    """Test gbasis.integrals.momentum.momentum_integral_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_mix(["spherical"] * 8, screen_basis=False),
        momentum_integral(basis, screen_basis=False),
    )


def test_momentum_integral_lincomb():
    """Test gbasis.integrals.momentum.momentum_integral_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    momentum_integral_obj = MomentumIntegral(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        momentum_integral_obj.construct_array_lincomb(transform, ["spherical"], screen_basis=False),
        momentum_integral(basis, transform=transform, screen_basis=False),
    )


@pytest.mark.parametrize("precision", [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
def test_momentum_screening_accuracy(precision):
    """Test momentum screening."""

    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    #  the screening tolerance needs to be 1e-4 times the desired precision
    tol_screen = precision * 1e-4
    momentum = momentum_integral(contraction, tol_screen=tol_screen)
    momentum_no_screen = momentum_integral(contraction, screen_basis=False)
    assert np.allclose(momentum, momentum_no_screen, atol=precision)
