"""Test gbasis.integrals.electron_repulsion."""
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._two_elec_int import (
    _compute_two_elec_integrals,
    _compute_two_elec_integrals_angmom_zero,
)
from gbasis.integrals.electron_repulsion import (
    electron_repulsion_integral,
    ElectronRepulsionIntegral,
)
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_construct_array_contraction():
    """Test integrals.electron_repulsion.ElectronRepulsionIntegral.construct_array_contraction."""
    coord_one = np.array([0.5, 1, 1.5])
    cont_one = GeneralizedContractionShell(
        0, coord_one, np.array([1.0]), np.array([0.1]), "spherical"
    )
    coord_two = np.array([1.5, 2, 3])
    cont_two = GeneralizedContractionShell(
        0, coord_two, np.array([3.0]), np.array([0.2]), "spherical"
    )
    coord_three = np.array([2.5, 3, 4])
    cont_three = GeneralizedContractionShell(
        0, coord_three, np.array([3.0]), np.array([0.2]), "spherical"
    )
    coord_four = np.array([3.5, 4, 5])
    cont_four = GeneralizedContractionShell(
        0, coord_four, np.array([3.0]), np.array([0.2]), "spherical"
    )

    with pytest.raises(TypeError):
        ElectronRepulsionIntegral.construct_array_contraction(None, cont_two, cont_three, cont_four)
    with pytest.raises(TypeError):
        ElectronRepulsionIntegral.construct_array_contraction(cont_one, None, cont_three, cont_four)
    with pytest.raises(TypeError):
        ElectronRepulsionIntegral.construct_array_contraction(cont_one, cont_two, None, cont_four)
    with pytest.raises(TypeError):
        ElectronRepulsionIntegral.construct_array_contraction(cont_one, cont_two, cont_three, None)

    integrals = _compute_two_elec_integrals_angmom_zero(
        ElectronRepulsionIntegral.boys_func,
        cont_one.coord,
        cont_one.exps,
        cont_one.coeffs,
        cont_two.coord,
        cont_two.exps,
        cont_two.coeffs,
        cont_three.coord,
        cont_three.exps,
        cont_three.coeffs,
        cont_four.coord,
        cont_four.exps,
        cont_four.coeffs,
    )
    integrals = np.transpose(integrals, (4, 0, 5, 1, 6, 2, 7, 3))
    assert np.allclose(
        integrals,
        ElectronRepulsionIntegral.construct_array_contraction(
            cont_one, cont_two, cont_three, cont_four
        ),
    )

    cont_four.angmom = 1
    integrals = _compute_two_elec_integrals(
        ElectronRepulsionIntegral.boys_func,
        cont_one.coord,
        cont_one.angmom,
        cont_one.angmom_components_cart,
        cont_one.exps,
        cont_one.coeffs,
        cont_two.coord,
        cont_two.angmom,
        cont_two.angmom_components_cart,
        cont_two.exps,
        cont_two.coeffs,
        cont_three.coord,
        cont_three.angmom,
        cont_three.angmom_components_cart,
        cont_three.exps,
        cont_three.coeffs,
        cont_four.coord,
        cont_four.angmom,
        cont_four.angmom_components_cart,
        cont_four.exps,
        cont_four.coeffs,
    )
    integrals = np.transpose(integrals, (4, 0, 5, 1, 6, 2, 7, 3))
    assert np.allclose(
        integrals,
        ElectronRepulsionIntegral.construct_array_contraction(
            cont_one, cont_two, cont_three, cont_four
        ),
    )


def test_electron_repulsion_cartesian_horton_sto6g_bec():
    """Test electron_repulsion.electron_repulsion_cartesian against horton results.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set STO-6G. Note
    that ano-rcc was not used because it results in overflow in the _compute_two_electron_integrals.

    """
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    coords = np.array([[0, 0, 0], [1.0, 0, 0]])
    basis = make_contractions(basis_dict, ["Be", "C"], coords, "cartesian")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_elec_repulsion = np.load(find_datafile("data_horton_bec_cart_elec_repulsion.npy"))
    assert np.allclose(horton_elec_repulsion, electron_repulsion_integral(basis))


def test_electron_repulsion_cartesian_horton_custom_hhe():
    """Test electron_repulsion.electron_repulsion_cartesian against horton results.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC
    modified. The basis set was modified to remove large exponent components to avoid overflow and
    some contractions for faster test.

    This test is also slow.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0], [0.8, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords, "cartesian")
    basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs[:, 0], i.exps, i.coord_type)
        for i in basis[:8]
    ]
    basis[0] = HortonContractions(
        basis[0].angmom, basis[0].coord, basis[0].coeffs[3:], basis[0].exps[3:], basis[0].coord_type
    )
    basis[4] = HortonContractions(
        basis[4].angmom, basis[4].coord, basis[4].coeffs[4:], basis[4].exps[4:], basis[4].coord_type
    )
    basis.pop(3)
    basis.pop(2)

    horton_elec_repulsion = np.load(find_datafile("data_horton_hhe_cart_elec_repulsion.npy"))
    assert np.allclose(horton_elec_repulsion, electron_repulsion_integral(basis))


def test_electron_repulsion_cartesian():
    """Test gbasis.integrals.electron_repulsion.electron_repulsion_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")

    erep_obj = ElectronRepulsionIntegral(basis)
    assert np.allclose(
        erep_obj.construct_array_cartesian(),
        electron_repulsion_integral(basis, notation="chemist"),
    )
    assert np.allclose(
        np.einsum("ijkl->ikjl", erep_obj.construct_array_cartesian()),
        electron_repulsion_integral(basis, notation="physicist"),
    )
    with pytest.raises(ValueError):
        electron_repulsion_integral(basis, notation="bad")


def test_electron_repulsion_spherical():
    """Test gbasis.integrals.electron_repulsion.electron_repulsion_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

    erep_obj = ElectronRepulsionIntegral(basis)
    assert np.allclose(
        erep_obj.construct_array_spherical(),
        electron_repulsion_integral(basis, notation="chemist"),
    )
    assert np.allclose(
        np.einsum("ijkl->ikjl", erep_obj.construct_array_spherical()),
        electron_repulsion_integral(basis, notation="physicist"),
    )
    with pytest.raises(ValueError):
        electron_repulsion_integral(basis, notation="bad")


def test_electron_repulsion_mix():
    """Test gbasis.integrals.electron_repulsion.electron_repulsion_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), ["spherical"] * 3)

    erep_obj = ElectronRepulsionIntegral(basis)
    assert np.allclose(
        erep_obj.construct_array_mix(["spherical"] * 3),
        electron_repulsion_integral(basis, notation="chemist"),
    )
    assert np.allclose(
        np.einsum("ijkl->ikjl", erep_obj.construct_array_mix(["spherical"] * 3)),
        electron_repulsion_integral(basis, notation="physicist"),
    )
    with pytest.raises(ValueError):
        electron_repulsion_integral(basis, notation="bad")


def test_electron_repulsion_lincomb():
    """Test gbasis.integrals.electron_repulsion.electron_repulsion_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

    erep_obj = ElectronRepulsionIntegral(basis)
    transform = np.random.rand(3, 5)
    assert np.allclose(
        erep_obj.construct_array_lincomb(transform, ["spherical"]),
        electron_repulsion_integral(basis, transform, notation="chemist"),
    )
    assert np.allclose(
        np.einsum("ijkl->ikjl", erep_obj.construct_array_lincomb(transform, ["spherical"])),
        electron_repulsion_integral(basis, transform, notation="physicist"),
    )
    with pytest.raises(ValueError):
        electron_repulsion_integral(basis, transform, notation="bad")
