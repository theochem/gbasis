"""Test gbasis.momentum."""
from gbasis._diff_operator_int import _compute_differential_operator_integrals
from gbasis.contractions import GeneralizedContractionShell
from gbasis.momentum import momentum_integral, MomentumIntegral
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_momentum_construct_array_contraction():
    """Test gbasis.momentum.MomentumIntegral.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02])
    )
    test = MomentumIntegral.construct_array_contraction(test_one, test_two).squeeze()
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
    """Test gbasis.momentum.momentum_integral_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_cartesian(),
        momentum_integral(basis, coord_type="cartesian"),
    )


def test_momentum_integral_spherical():
    """Test gbasis.momentum.momentum_integral_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_spherical(),
        momentum_integral(basis, coord_type="spherical"),
    )


def test_momentum_integral_mix():
    """Test gbasis.momentum.momentum_integral_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    momentum_integral_obj = MomentumIntegral(basis)
    assert np.allclose(
        momentum_integral_obj.construct_array_mix(["spherical"] * 8),
        momentum_integral(basis, coord_type=["spherical"] * 8),
    )


def test_momentum_integral_lincomb():
    """Test gbasis.momentum.momentum_integral_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    momentum_integral_obj = MomentumIntegral(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        momentum_integral_obj.construct_array_lincomb(transform, "spherical"),
        momentum_integral(basis, transform=transform, coord_type="spherical"),
    )
