"""Test gbasis.integrals.kinetic_energy."""
import numpy as np
import pytest
from utils import HortonContractions, find_datafile

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import _compute_differential_operator_integrals
from gbasis.integrals.kinetic_energy import KineticEnergyIntegral, kinetic_energy_integral
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.utils import factorial2


def test_kinetic_energy_construct_array_contraction():
    """Test gbasis.integrals.kinetic_energy.KineticEnergyIntegral.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02]), "spherical"
    )
    answer = np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[2, 0, 0]]),
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
    answer += np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[0, 2, 0]]),
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
    answer += np.array(
        [
            [
                _compute_differential_operator_integrals(
                    np.array([[0, 0, 2]]),
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
    assert np.allclose(
        np.squeeze(KineticEnergyIntegral.construct_array_contraction(test_one, test_two)),
        np.squeeze(-0.5 * answer),
    )

    with pytest.raises(TypeError):
        KineticEnergyIntegral.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        KineticEnergyIntegral.construct_array_contraction(None, test_two)


def test_kinetic_energy_integral_cartesian():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_cartesian(),
        kinetic_energy_integral(basis),
    )


def test_kinetic_energy_integral_spherical():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_spherical(),
        kinetic_energy_integral(basis),
    )


def test_kinetic_energy_integral_mix():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_mix(["spherical"] * 8),
        kinetic_energy_integral(basis),
    )


def test_kinetic_energy_integral_lincomb():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_lincomb(transform, ["spherical"]),
        kinetic_energy_integral(basis, transform),
    )


def test_kinetic_energy_integral_horton_anorcc_hhe():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian against HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict,
        ["H", "He"],
        np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]]),
        "cartesian",
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_kinetic_energy_integral = np.load(
        find_datafile("data_horton_hhe_cart_kinetic_energy_integral.npy")
    )
    assert np.allclose(kinetic_energy_integral(basis), horton_kinetic_energy_integral)


def test_kinetic_energy_integral_horton_anorcc_bec():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian against HORTON.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict,
        ["Be", "C"],
        np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]]),
        "cartesian",
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_kinetic_energy_integral = np.load(
        find_datafile("data_horton_bec_cart_kinetic_energy_integral.npy")
    )
    assert np.allclose(kinetic_energy_integral(basis), horton_kinetic_energy_integral)
