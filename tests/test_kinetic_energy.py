"""Test gbasis.integrals.kinetic_energy."""
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import _compute_differential_operator_integrals
from gbasis.integrals.kinetic_energy import kinetic_energy_integral, KineticEnergyIntegral
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
from gbasis.utils import factorial2
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


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

    val = KineticEnergyIntegral.construct_array_contraction(test_one, test_two, screen_basis=False)
    assert np.allclose(np.squeeze(val), np.squeeze(-0.5 * answer))

    with pytest.raises(TypeError):
        KineticEnergyIntegral.construct_array_contraction(test_one, None, screen_basis=False)
    with pytest.raises(TypeError):
        KineticEnergyIntegral.construct_array_contraction(None, test_two, screen_basis=False)


def test_kinetic_energy_integral_cartesian():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_cartesian(screen_basis=False),
        kinetic_energy_integral(basis, screen_basis=False),
    )


def test_kinetic_energy_integral_spherical():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_spherical(screen_basis=False),
        kinetic_energy_integral(basis, screen_basis=False),
    )


def test_kinetic_energy_integral_mix():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_mix(["spherical"] * 8, screen_basis=False),
        kinetic_energy_integral(basis, screen_basis=False),
    )


def test_kinetic_energy_integral_lincomb():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    kinetic_energy_integral_obj = KineticEnergyIntegral(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        kinetic_energy_integral_obj.construct_array_lincomb(
            transform, ["spherical"] * 8, screen_basis=False
        ),
        kinetic_energy_integral(basis, transform, screen_basis=False),
    )


def test_kinetic_energy_integral_horton_anorcc_hhe():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian against HORTON's results.

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
    ke_integral = kinetic_energy_integral(basis, screen_basis=False)
    assert np.allclose(ke_integral, horton_kinetic_energy_integral)


def test_kinetic_energy_integral_horton_anorcc_bec():
    """Test gbasis.integrals.kinetic_energy.kinetic_energy_integral_cartesian against HORTON's results.

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
    ke_int_values = kinetic_energy_integral(basis, screen_basis=False)
    assert np.allclose(ke_int_values, horton_kinetic_energy_integral)

@pytest.mark.parametrize("precision", [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
def test_kinetic_screening_accuracy(precision):
    """Test kinetic energy screening.

    This test is meant to  fail.  Using cartesian sto-6G nwchem basis set.

    """
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    #  the screening tolerance needs to be 1e-4 times the desired precision
    tol_screen = precision * 1e-4
    kinetic_energy = kinetic_energy_integral(contraction, tol_screen=tol_screen)
    kinetic_energy_no_screen = kinetic_energy_integral(contraction, screen_basis=False)
    assert np.allclose(kinetic_energy, kinetic_energy_no_screen, atol=precision)
