"""Test gbasis.integrals.nuclear_electron_attraction."""

from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.point_charge import point_charge_integral
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
from utils import find_datafile, HortonContractions
import pytest


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_horton_anorcc_hhe(screen_basis, tol_screen):
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords, "cartesian")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_integral(basis, coords, np.array([1, 2]), **kwargs),
        horton_nucattract,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_horton_anorcc_bec(screen_basis, tol_screen):
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with B and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["Be", "C"], coords, "cartesian")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_bec_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_integral(basis, coords, np.array([4, 6]), **kwargs),
        horton_nucattract,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_cartesian(screen_basis, tol_screen):
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_cartesian."""
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, **kwargs)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(basis, nuclear_coords, nuclear_charges, **kwargs),
        atol=1e-10,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_spherical(screen_basis, tol_screen):
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_spherical."""
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, **kwargs)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(basis, nuclear_coords, nuclear_charges, **kwargs),
        atol=1e-10,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_mix(screen_basis, tol_screen):
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_mix."""
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, **kwargs)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(basis, nuclear_coords, nuclear_charges, **kwargs),
        atol=1e-10,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_nuclear_electron_attraction_lincomb(screen_basis, tol_screen):
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_lincomb."""
    kwargs = {"screen_basis": screen_basis, "tol_screen": tol_screen}
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    transform = np.random.rand(14, 18)
    ref = point_charge_integral(
        basis, nuclear_coords, nuclear_charges, transform=transform, **kwargs
    )
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(
            basis, nuclear_coords, nuclear_charges, transform=transform, **kwargs
        ),
        atol=1e-10,
    )
