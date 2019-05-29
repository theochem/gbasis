"""Test gbasis.nuclear_electron_attraction."""
from gbasis.nuclear_electron_attraction import (
    nuclear_electron_attraction_cartesian,
    nuclear_electron_attraction_lincomb,
    nuclear_electron_attraction_mix,
    nuclear_electron_attraction_spherical,
)
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.point_charge import (
    point_charge_cartesian,
    point_charge_lincomb,
    point_charge_mix,
    point_charge_spherical,
)
import numpy as np
from utils import find_datafile, HortonContractions


def test_nuclear_electron_attraction_horton_anorcc_hhe():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_cartesian(basis, coords, np.array([1, 2])), horton_nucattract
    )


def test_nuclear_electron_attraction_horton_anorcc_bec():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with B and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["Be", "C"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_bec_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_cartesian(basis, coords, np.array([4, 6])), horton_nucattract
    )


def test_nuclear_electron_attraction_cartesian():
    """Test gbasis.nuclear_electron_attraction.nuclear_electron_attraction_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_cartesian(basis, nuclear_coords, nuclear_charges)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_cartesian(basis, nuclear_coords, nuclear_charges),
    )


def test_nuclear_electron_attraction_spherical():
    """Test gbasis.nuclear_electron_attraction.nuclear_electron_attraction_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_spherical(basis, nuclear_coords, nuclear_charges)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_spherical(basis, nuclear_coords, nuclear_charges),
    )


def test_nuclear_electron_attraction_mix():
    """Test gbasis.nuclear_electron_attraction.nuclear_electron_attraction_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_mix(basis, nuclear_coords, nuclear_charges, ["spherical"] * 8)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_mix(basis, nuclear_coords, nuclear_charges, ["spherical"] * 8),
    )


def test_nuclear_electron_attraction_lincomb():
    """Test gbasis.nuclear_electron_attraction.nuclear_electron_attraction_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    transform = np.random.rand(14, 18)
    ref = point_charge_lincomb(basis, transform, nuclear_coords, nuclear_charges)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_lincomb(basis, transform, nuclear_coords, nuclear_charges),
    )
