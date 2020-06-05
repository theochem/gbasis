"""Test gbasis.integrals.overlap."""
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals
from gbasis.integrals.overlap import Overlap, overlap_integral
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile, HortonContractions


def test_overlap_construct_array_contraction():
    """Test gbasis.integrals.overlap.Overlap.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02])
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
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(
        overlap_obj.construct_array_cartesian(), overlap_integral(basis, coord_type="cartesian")
    )


def test_overlap_spherical():
    """Test gbasis.integrals.overlap.overlap_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(
        overlap_obj.construct_array_spherical(), overlap_integral(basis, coord_type="spherical")
    )


def test_overlap_mix():
    """Test gbasis.integrals.overlap.overlap_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(
        overlap_obj.construct_array_mix(["spherical"] * 8),
        overlap_integral(basis, coord_type=["spherical"] * 8),
    )


def test_overlap_lincomb():
    """Test gbasis.integrals.overlap.overlap_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        overlap_obj.construct_array_lincomb(transform, "spherical"),
        overlap_integral(basis, transform=transform, coord_type="spherical"),
    )


def test_overlap_cartesian_norm_anorcc():
    """Test the norm of gbasis.integrals.overlap_cartesian on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the cartesian contractions are normalized.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_spherical_norm_sto6g():
    """Test the norm of gbasis.integrals.overlap_spherical on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the spherical contractions are not
    normalized to past 3rd decimal places.

    """
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_spherical()), 1)


def test_overlap_spherical_norm_anorcc():
    """Test the norm of gbasis.integrals.overlap_spherical on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the Cartesian contractions are normalized.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))

    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)

    basis = make_contractions(basis_dict, ["Xe"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_cartesian_norm_sto6g():
    """Test the norm of gbasis.integrals.overlap_cartesian on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the Cartesian contractions are not
    normalized to past 3rd decimal places.

    """
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_horton_anorcc_hhe():
    """Test gbasis.integrals.overlap.overlap_basis_cartesian against HORTON's overlap matrix.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["H", "He"], np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_overlap = np.load(find_datafile("data_horton_hhe_cart_overlap.npy"))
    assert np.allclose(overlap_integral(basis, coord_type="cartesian"), horton_overlap)


def test_overlap_horton_anorcc_bec():
    """Test gbasis.integrals.overlap.overlap_cartesian against HORTON's overlap matrix.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["Be", "C"], np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_overlap = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(overlap_integral(basis, coord_type="cartesian"), horton_overlap)


def test_overlap_screening_631g_gbs():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))

    # Test 1
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    # tolerance for overlap is by default 1E-20. To change add tol = X as the end argument
    contraction[0].create_overlap_mask(contraction)
    if not any(contraction[0].ovr_mask):
        raise ValueError("Two hydrogens at this distance should require overlap calculations")

    # Test 2
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 20]]))
    alpha_a = alpha_b = min(contraction[0].exps)
    rij = np.linalg.norm(np.array([0, 0, 0]) - np.array([1, 1, 10]))
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(1e-20))
    # tolerance for overlap is by default 1e-20. To change add tol = X to make_contractions()
    contraction[0].create_overlap_mask(contraction)
    if contraction[0].ovr_mask[2] or contraction[0].ovr_mask[2]:
        raise ValueError(
            "Two hydrogens at this distance should NOT require overlap calculations"
            " distance is {} and cutoff distance is {}".format(rij, cutoff)
        )
    if not contraction[0].ovr_mask[0] or not contraction[0].ovr_mask[1]:
        raise ValueError("Self overlaps should not be skipped in overlap screening")


def test_overlap_screening_sto6g_gbs():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_gbs(find_datafile("data_sto6g.gbs"))

    # Test 1
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    contraction[0].create_overlap_mask(contraction)
    if not any(contraction[0].ovr_mask):
        raise ValueError("Two hydrogens at this distance should require overlap calculations")

    # Test 2
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 31]]))
    alpha_a = alpha_b = min(contraction[0].exps)
    rij = np.linalg.norm(np.array([0, 0, 0]) - np.array([1, 1, 10]))
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(1e-20))
    # tolerance for overlap is by default 1e-20. To change add tol = X to make_contractions()
    contraction[0].create_overlap_mask(contraction)
    if contraction[0].ovr_mask[1] or contraction[0].ovr_mask[1]:
        raise ValueError(
            "Two hydrogens at this distance should NOT require overlap calculations."
            " Distance is {} and cutoff distance is {}".format(rij, cutoff)
        )
    # Test self interactions between contractions.
    contraction[1].create_overlap_mask(contraction)
    if not contraction[0].ovr_mask[0] or not contraction[1].ovr_mask[1]:
        raise ValueError("Self overlaps should not be skipped in overlap screening")


def test_overlap_screening_631g_nwchem():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_nwchem(find_datafile("data_631g.nwchem"))

    # Test 1
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    contraction[0].create_overlap_mask(contraction)
    if not any(contraction[0].ovr_mask):
        raise ValueError("Two hydrogens at this distance should require overlap calculations")

    # Test 2
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 20]]))
    alpha_a = alpha_b = min(contraction[0].exps)
    rij = np.linalg.norm(np.array([0, 0, 0]) - np.array([1, 1, 10]))
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(1.0e-20))
    # tolerance for overlap is by default 1e-20. To change add tol = X to make_contractions()
    contraction[0].create_overlap_mask(contraction)
    if contraction[0].ovr_mask[2] or contraction[0].ovr_mask[2]:
        raise ValueError(
            "Two hydrogens at this distance should NOT require overlap calculations"
            " distance is {} and cutoff distance is {}".format(rij, cutoff)
        )
    if not contraction[0].ovr_mask[0] or not contraction[0].ovr_mask[1]:
        raise ValueError("Self overlaps should not be skipped in overlap screening")


def test_overlap_screening_sto6g_nwchem():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    # Test 1
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    # tolerance for overlap is by default 1E-20. To change add tol = X as the end argument
    contraction[0].create_overlap_mask(contraction)
    if not any(contraction[0].ovr_mask):
        raise ValueError("Two hydrogens at this distance should require overlap calculations")

    # Test 2
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 31]]))
    alpha_a = alpha_b = min(contraction[0].exps)
    rij = np.linalg.norm(np.array([0, 0, 0]) - np.array([1, 1, 10]))
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(1.0e-20))
    # tolerance for overlap is by default 1E-20. To change add tol = X to make_contractions()
    contraction[0].create_overlap_mask(contraction)
    if contraction[0].ovr_mask[1] or contraction[0].ovr_mask[1]:
        raise ValueError(
            "Two hydrogens at this distance should NOT require overlap calculations."
            " Distance is {} and cutoff distance is {}".format(rij, cutoff)
        )
    # Test self interactions between contractions.
    contraction[1].create_overlap_mask(contraction)
    if not contraction[0].ovr_mask[0] or not contraction[1].ovr_mask[1]:
        raise ValueError("Self overlaps should not be skipped in overlap screening")


def test_overlap_screening_vs_without_screening():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    # Test 1
    contraction = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    contraction_without_screen = make_contractions(
        basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), overlap=False
    )
    # tolerance for overlap is by default 1E-20. To change add tol = X as the end argument
    contraction[0].create_overlap_mask(contraction)
    contraction[1].create_overlap_mask(contraction)

    overlaps = overlap_integral(contraction)
    overlaps_no_screen = overlap_integral(contraction_without_screen)

    assert np.allclose(overlaps, overlaps_no_screen)

    # Test that screened and unscreened overlaps are the SAME with tight tolerances
    contraction = make_contractions(
        basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 31]])
    )
    contraction_without_screen = make_contractions(
        basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 31]]), overlap=False
    )

    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    for i in range(len(contraction)):
        contraction[i].create_overlap_mask(contraction)

    overlaps = overlap_integral(contraction)
    overlaps_no_screen = overlap_integral(contraction_without_screen)

    assert np.allclose(overlaps, overlaps_no_screen, rtol=1.0e-5, atol=1.0e-8)

    # Test that screened and un-screened overlaps are DIFFERENT [use large tol for (small) cutoff]
    contraction = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]])
    )
    contraction_without_screen = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]])
    )
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    for item in contraction:
        item.create_overlap_mask(contraction, tol=1.0e-4)

    overlaps = overlap_integral(contraction)
    overlaps_no_screen = overlap_integral(contraction_without_screen)

    assert not np.allclose(overlaps, overlaps_no_screen, rtol=1.0e-5, atol=1.0e-8)

    # Use gaussian file now, and also switch to using 6-31G basis set
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    # Test that screened and unscreened overlaps are the SAME with very tight tolerances
    contraction = make_contractions(
        basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 31]])
    )
    contraction_without_screen = make_contractions(
        basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 31]]), overlap=False
    )
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    for item in contraction:
        item.create_overlap_mask(contraction)

    overlaps = overlap_integral(contraction)
    overlaps_no_screen = overlap_integral(contraction_without_screen)

    assert np.allclose(overlaps, overlaps_no_screen, rtol=1.0e-5, atol=1.0e-8)

    # Test that screened and unscreened overlaps are DIFFERENT [use large tol for (small) cutoff]
    contraction = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]])
    )
    contraction_without_screen = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]]), overlap=False
    )
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    for item in contraction:
        item.create_overlap_mask(contraction, tol=1.0e-4)

    overlaps = overlap_integral(contraction)
    overlaps_no_screen = overlap_integral(contraction_without_screen)

    assert not np.allclose(overlaps, overlaps_no_screen, rtol=1.0e-5, atol=1.0e-8)


def test_overlap_screening_with_cartesian():
    """Test that the overlap screening mask has been correctly created for several cases."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    # Test that screened and unscreened overlaps are DIFFERENT [use large tol for (small) cutoff]
    # Note #2 overlap=True tests the creation of masks in parsers. Do not remove!
    contraction = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]]), overlap=True
    )
    contraction_without_screen = make_contractions(
        basis_dict, ["C", "C", "C"], np.array([[0, 0, 0], [0, 0, 11], [1, 1, 20]]), overlap=False
    )
    # tolerance for overlap is by default 1e-20. To change add tol = X as the end argument
    for item in contraction:
        item.create_overlap_mask(contraction, tol=1.0e-4)

    overlaps = overlap_integral(contraction, coord_type="cartesian")
    overlaps_no_screen = overlap_integral(contraction_without_screen, coord_type="cartesian")

    assert not np.allclose(overlaps, overlaps_no_screen, rtol=1.0e-5, atol=1.0e-8)
