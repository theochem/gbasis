"""Test gbasis.integrals.overlap_asymm."""
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric
from gbasis.parsers import make_contractions, parse_nwchem, parse_gbs
import numpy as np
from utils import find_datafile, HortonContractions
import pytest


def test_overlap_integral_asymmetric_horton_anorcc_hhe():
    """Test gbasis.integrals.overlap_asymm.overlap_integral_asymmetric against HORTON's overlap matrix.

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

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_hhe_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(basis, basis, screen_basis=False),
        horton_overlap_integral_asymmetric,
    )


def test_overlap_integral_asymmetric_horton_anorcc_bec():
    """Test integrals.overlap_asymm.overlap_integral_asymmetric against HORTON's overlap matrix.

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

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(basis, basis, screen_basis=False),
        horton_overlap_integral_asymmetric,
    )


def test_overlap_integral_asymmetric_compare():
    """Test overlap_asymm.overlap_integral_asymmetric against overlap.overlap_integral."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    cartesian_basis = make_contractions(
        basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]), "cartesian"
    )
    spherical_basis = make_contractions(
        basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]), "spherical"
    )
    mixed_basis = make_contractions(
        basis_dict,
        ["Kr", "Kr"],
        np.array([[0, 0, 0], [1.0, 0, 0]]),
        ["spherical"] * 9 + ["cartesian"],
    )
    cartesian_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in cartesian_basis
    ]
    spherical_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in spherical_basis
    ]
    mixed_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in mixed_basis
    ]

    assert np.allclose(
        overlap_integral(cartesian_basis, screen_basis=False),
        overlap_integral_asymmetric(cartesian_basis, cartesian_basis, screen_basis=False),
    )
    assert np.allclose(
        overlap_integral(spherical_basis, screen_basis=False),
        overlap_integral_asymmetric(spherical_basis, spherical_basis, screen_basis=False),
    )
    assert np.allclose(
        overlap_integral(spherical_basis, transform=np.identity(218), screen_basis=False),
        overlap_integral_asymmetric(
            spherical_basis,
            spherical_basis,
            transform_one=np.identity(218),
            transform_two=np.identity(218),
            screen_basis=False,
        ),
    )
    assert np.allclose(
        overlap_integral(mixed_basis, screen_basis=False),
        overlap_integral_asymmetric(
            mixed_basis,
            mixed_basis,
            screen_basis=False,
        ),
    )


@pytest.mark.parametrize("precision", [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
def test_overlap_asymmetric_screening_accuracy(precision):
    """Test asymmetric overlap screening."""

    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    #  the screening tolerance needs to be 1e-4 times the desired precision
    tol_screen = precision * 1e-4
    overlap_asymmetric = overlap_integral_asymmetric(contraction, contraction, tol_screen=tol_screen)
    overlap_asymmetric_no_screen = overlap_integral_asymmetric(contraction, contraction, screen_basis=False)
    assert np.allclose(overlap_asymmetric, overlap_asymmetric_no_screen, atol=precision)
