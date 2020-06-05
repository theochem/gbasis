"""Test gbasis.integrals.overlap_asymm."""
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
from utils import find_datafile, HortonContractions


def test_overlap_integral_asymmetric_horton_anorcc_hhe():
    """Test gbasis.integrals.overlap_asymm.overlap_integral_asymmetric against HORTON's overlap matrix.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["H", "He"], np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_hhe_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(
            basis, basis, coord_type_two="cartesian", coord_type_one="cartesian"
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
        basis_dict, ["Be", "C"], np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_overlap_integral_asymmetric = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(
        overlap_integral_asymmetric(
            basis, basis, coord_type_two="cartesian", coord_type_one="cartesian"
        ),
        horton_overlap_integral_asymmetric,
    )


def test_overlap_integral_asymmetric_horton_anorcc_bec_screening_passes():
    """Test integrals.overlap_asymm.overlap_integral_asymmetric for screened and unscreened overlap.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis_screened = make_contractions(
        basis_dict,
        ["Be", "C"],
        np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]]),
        overlap=True,
    )
    basis_unscreened = make_contractions(
        basis_dict,
        ["Be", "C"],
        np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]]),
        overlap=False,
    )

    overlap_screened = overlap_integral(basis_screened)
    overlap_unscreened = overlap_integral(basis_unscreened)

    assert np.allclose(overlap_screened, overlap_unscreened)


def test_overlap_integral_asymmetric_horton_anorcc_bec_screening_fails():
    """Test integrals.overlap_asymm.overlap_integral_asymmetric for screened and unscreened overlap.

    The test case is diatomic with Be and C with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: atoms are further apart than previous test. Don't use overlap=True. See below.
    basis_screened = make_contractions(
        basis_dict, ["Be", "C"], np.array([[0, 0, 0], [10.0 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis_unscreened = make_contractions(
        basis_dict,
        ["Be", "C"],
        np.array([[0, 0, 0], [10.0 * 1.0 / 0.5291772083, 0, 0]]),
        overlap=False,
    )
    # manually make masks so that custom tolerance can be used
    for i in range(len(basis_screened)):
        basis_screened[i].create_overlap_mask(basis_screened, tol=0.001)

    overlap_screened = overlap_integral(basis_screened)
    overlap_unscreened = overlap_integral(basis_unscreened)

    # np.allclose() should say they are not the same given the very large value for tol.
    assert not np.allclose(overlap_screened, overlap_unscreened)


def test_overlap_integral_asymmetric_compare():
    """Test overlap_asymm.overlap_integral_asymmetric against overlap.overlap_integral."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))

    basis = make_contractions(basis_dict, ["Kr", "Kr"], np.array([[0, 0, 0], [1.0, 0, 0]]))
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    assert np.allclose(
        overlap_integral(basis, coord_type="cartesian"),
        overlap_integral_asymmetric(
            basis, basis, coord_type_one="cartesian", coord_type_two="cartesian"
        ),
    )
    assert np.allclose(
        overlap_integral(basis, coord_type="spherical"),
        overlap_integral_asymmetric(
            basis, basis, coord_type_one="spherical", coord_type_two="spherical"
        ),
    )
    assert np.allclose(
        overlap_integral(basis, transform=np.identity(218), coord_type="spherical"),
        overlap_integral_asymmetric(
            basis,
            basis,
            transform_one=np.identity(218),
            transform_two=np.identity(218),
            coord_type_one="spherical",
            coord_type_two="spherical",
        ),
    )
    assert np.allclose(
        overlap_integral(basis, coord_type=["spherical"] * 9 + ["cartesian"]),
        overlap_integral_asymmetric(
            basis,
            basis,
            coord_type_one=["spherical"] * 9 + ["cartesian"],
            coord_type_two=["spherical"] * 9 + ["cartesian"],
        ),
    )
