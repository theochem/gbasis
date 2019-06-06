"""Test gbasis.integrals.nuclear_electron_attraction."""
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.point_charge import point_charge_integral
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
from utils import find_datafile, HortonContractions


def test_nuclear_electron_attraction_horton_anorcc_hhe():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_integral(
            basis, coords, np.array([1, 2]), coord_type="cartesian"
        ),
        horton_nucattract,
    )


def test_nuclear_electron_attraction_horton_anorcc_bec():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_cartesian with HORTON.

    The test case is diatomic with B and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["Be", "C"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_bec_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_integral(
            basis, coords, np.array([4, 6]), coord_type="cartesian"
        ),
        horton_nucattract,
    )


def test_nuclear_electron_attraction_cartesian():
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, coord_type="cartesian")
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(
            basis, nuclear_coords, nuclear_charges, coord_type="cartesian"
        ),
    )


def test_nuclear_electron_attraction_spherical():
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, coord_type="spherical")
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(
            basis, nuclear_coords, nuclear_charges, coord_type="spherical"
        ),
    )


def test_nuclear_electron_attraction_mix():
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    ref = point_charge_integral(
        basis, nuclear_coords, nuclear_charges, coord_type=["spherical"] * 8
    )
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(
            basis, nuclear_coords, nuclear_charges, coord_type=["spherical"] * 8
        ),
    )


def test_nuclear_electron_attraction_lincomb():
    """Test gbasis.integrals.nuclear_electron_attraction.nuclear_electron_attraction_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))

    nuclear_coords = np.random.rand(5, 3)
    nuclear_charges = np.random.rand(5)
    transform = np.random.rand(14, 18)
    ref = point_charge_integral(basis, nuclear_coords, nuclear_charges, transform=transform)
    assert np.allclose(
        ref[:, :, 0] + ref[:, :, 1] + ref[:, :, 2] + ref[:, :, 3] + ref[:, :, 4],
        nuclear_electron_attraction_integral(
            basis, nuclear_coords, nuclear_charges, transform=transform
        ),
    )
