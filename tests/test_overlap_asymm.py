"""Test gbasis.integrals.overlap_asymm."""
import numpy as np
from utils import HortonContractions, find_datafile

from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric
from gbasis.parsers import make_contractions, parse_nwchem


def test_overlap_integral_asymmetric_horton_anorcc_hhe():
    """Test gbasis.integrals.overlap_asymm.overlap_integral_asymmetric against HORTON overlap mat.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["H", "He"], np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]]), "cartesian"
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_hhe_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(
            basis, basis
        ),
        horton_overlap_integral_asymmetric,
    )


def test_overlap_integral_asymmetric_horton_anorcc_bec():
    """Test integrals.overlap_asymm.overlap_integral_asymmetric against HORTON's overlap matrix.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["Be", "C"], np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]]), "cartesian"
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(
            basis, basis
        ),
        horton_overlap_integral_asymmetric,
    )


def test_overlap_integral_asymmetric_compare():
    """Test overlap_asymm.overlap_integral_asymmetric against overlap.overlap_integral."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    cartesian_basis = make_contractions(basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]), "cartesian")
    spherical_basis = make_contractions(basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]), "spherical")
    mixed_basis = make_contractions(basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]), ['spherical'] * 9 + ['cartesian'])
    cartesian_basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in cartesian_basis]
    spherical_basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in spherical_basis]
    mixed_basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in mixed_basis]

    assert np.allclose(
        overlap_integral(cartesian_basis),
        overlap_integral_asymmetric(
            cartesian_basis, cartesian_basis
        ),
    )
    assert np.allclose(
        overlap_integral(spherical_basis),
        overlap_integral_asymmetric(
            spherical_basis, spherical_basis
        ),
    )
    assert np.allclose(
        overlap_integral(spherical_basis, transform=np.identity(218)),
        overlap_integral_asymmetric(
            spherical_basis,
            spherical_basis,
            transform_one=np.identity(218),
            transform_two=np.identity(218),
        ),
    )
    assert np.allclose(
        overlap_integral(mixed_basis),
        overlap_integral_asymmetric(
            mixed_basis,
            mixed_basis,
        ),
    )
