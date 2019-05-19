"""Test gbasis.nuclear_electron_attraction."""
from gbasis.contractions import make_contractions
from gbasis.nuclear_electron_attraction import nuclear_electron_attraction_gbasis_cartesian
from gbasis.parsers import parse_nwchem
import numpy as np
from utils import find_datafile, HortonContractions


def test_nuclear_electron_attraction_horton_anorcc_hhe():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_gbasis_cartesian with HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.charge, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_gbasis_cartesian(basis, coords, np.array([1, 2])),
        horton_nucattract,
    )


def test_nuclear_electron_attraction_horton_anorcc_bec():
    """Test nuclear_electron_attraciton.nuclear_electron_attraction_gbasis_cartesian with HORTON.

    The test case is diatomic with B and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["Be", "C"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.charge, i.coeffs, i.exps) for i in basis]

    horton_nucattract = np.load(find_datafile("data_horton_bec_cart_nucattract.npy"))
    assert np.allclose(
        nuclear_electron_attraction_gbasis_cartesian(basis, coords, np.array([4, 6])),
        horton_nucattract,
    )
