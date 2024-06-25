"""Test gbasis.integrals.overlap."""

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals
from gbasis.integrals.overlap import Overlap, overlap_integral
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
from gbasis.utils import factorial2
import numpy as np
import pytest

from utils import find_datafile, HortonContractions


def test_overlap_construct_array_contraction():
    """Test gbasis.integrals.overlap.Overlap.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02]), "spherical"
    )
    answer = np.array(
        [
            [
                _compute_multipole_moment_integrals(
                    np.array([0, 0, 0]),
                    np.array([[0, 0, 0]]),
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
        np.squeeze(Overlap.construct_array_contraction(test_one, test_two)), np.squeeze(answer)
    )

    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(None, test_two)


def test_overlap_cartesian():
    """Test gbasis.integrals.overlap.overlap_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    overlap_obj = Overlap(basis)
    assert np.allclose(overlap_obj.construct_array_cartesian(), overlap_integral(basis))


def test_overlap_spherical():
    """Test gbasis.integrals.overlap.overlap_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    overlap_obj = Overlap(basis)
    assert np.allclose(overlap_obj.construct_array_spherical(), overlap_integral(basis))


def test_overlap_mix():
    """Test gbasis.integrals.overlap.overlap_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)
    overlap_obj = Overlap(basis)
    assert np.allclose(
        overlap_obj.construct_array_mix(["spherical"] * 8),
        overlap_integral(basis, tol_screen=None),
    )


def test_overlap_lincomb():
    """Test gbasis.integrals.overlap.overlap_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    overlap_obj = Overlap(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        overlap_obj.construct_array_lincomb(transform, ["spherical"]),
        overlap_integral(basis, transform=transform),
    )


def test_overlap_cartesian_norm_anorcc():
    """Test the norm of gbasis.integrals.overlap_cartesian on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the cartesian contractions are normalized.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_spherical_norm_sto6g():
    """Test the norm of gbasis.integrals.overlap_spherical on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the spherical contractions are not
    normalized to past 3rd decimal places.

    """
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_spherical()), 1)


def test_overlap_spherical_norm_anorcc():
    """Test the norm of gbasis.integrals.overlap_spherical on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the Cartesian contractions are normalized.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))

    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)

    basis = make_contractions(basis_dict, ["Xe"], np.array([[0, 0, 0]]), "cartesian")
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_cartesian_norm_sto6g():
    """Test the norm of gbasis.integrals.overlap_cartesian on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the Cartesian contractions are not
    normalized to past 3rd decimal places.

    """
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_horton_anorcc_hhe():
    """Test gbasis.integrals.overlap.overlap_basis_cartesian against HORTON's overlap matrix.

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

    horton_overlap = np.load(find_datafile("data_horton_hhe_cart_overlap.npy"))
    assert np.allclose(overlap_integral(basis), horton_overlap)


def test_overlap_horton_anorcc_bec():
    """Test gbasis.integrals.overlap.overlap_cartesian against HORTON's overlap matrix.

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

    horton_overlap = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(overlap_integral(basis), horton_overlap)


def test_overlap_screening_vs_without_screening():
    """Test overlap screening.

    This test is meant to  pass.  Using spherical 6-31G Gaussian basis set.

    """
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "spherical")

    # check overlap integrals with and without screening match
    overlaps = overlap_integral(contraction, tol_screen=1.0e-10)
    overlaps_no_screen = overlap_integral(contraction, tol_screen=None)
    assert np.allclose(overlaps, overlaps_no_screen)


def test_overlap_screening_with_fail():
    """Test overlap screening.

    This test is meant to  fail.  Using cartesian sto-6G nwchem basis set.

    """
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    # check overlap integrals with and without screening do not match
    overlaps = overlap_integral(contraction, tol_screen=1.0e-2)
    overlaps_no_screen = overlap_integral(contraction, tol_screen=None)
    assert not np.allclose(overlaps, overlaps_no_screen)
