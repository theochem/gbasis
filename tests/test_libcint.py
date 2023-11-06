"""Test gbasis.integrals.libcint."""

import pytest

import numpy as np

from gbasis.integrals.overlap import overlap_integral

from gbasis.integrals.libcint import CBasis

from gbasis.parsers import make_contractions, parse_nwchem

from utils import HortonContractions, find_datafile


def test_overlap_ugbs_hhe():
    """Test gbasis.integrals.libcint.CBasis.olp against GBasis's overlap matrix.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set UGBS.

    """
    basis_dict = parse_nwchem(find_datafile("data_ugbs.nwchem"))

    atsyms = ["H", "He"]

    atcoords = np.array([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083

    gbasis = make_contractions(basis_dict, atsyms, atcoords)
    gbasis_olp = overlap_integral(gbasis, coord_type="cartesian")

    cbasis = CBasis(gbasis, atsyms, atcoords, coord_type="cartesian")
    cbasis_olp = cbasis.olp()

    assert cbasis_olp.shape[0] == cbasis_olp.shape[1] == \
           gbasis_olp.shape[0] == gbasis_olp.shape[1] == \
           cbasis.nbas

    print(cbasis_olp)

    assert np.allclose(np.diag(cbasis_olp), 1)
    assert np.allclose(cbasis_olp, gbasis_olp)


def test_overlap_horton_ugbs_bec():
    """Test gbasis.integrals.libcint.CBasis.olp against GBasis's overlap matrix.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set UGBS.

    """
    basis_dict = parse_nwchem(find_datafile("data_ugbs.nwchem"))

    atsyms = ["Be", "C"]

    atcoords = np.array([[0., 0., 0.], [1.0, 0., 0.]]) / 0.5291772083

    gbasis = make_contractions(basis_dict, atsyms, atcoords)
    gbasis_olp = overlap_integral(gbasis, coord_type="cartesian")

    cbasis = CBasis(gbasis, atsyms, atcoords, coord_type="cartesian")
    cbasis_olp = cbasis.olp()

    assert cbasis_olp.shape[0] == cbasis_olp.shape[1] == \
           gbasis_olp.shape[0] == gbasis_olp.shape[1] == \
           cbasis.nbas

    print(cbasis_olp)

    assert np.allclose(np.diag(cbasis_olp), 1)
    assert np.allclose(cbasis_olp, gbasis_olp)


# def test_overlap_anorcc_hhe():
#     """Test gbasis.integrals.libcint.CBasis.olp against GBasis's overlap matrix.
#
#     The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
#
#     """
#     basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
#
#     atsyms = ["H", "He"]
#
#     atcoords = np.array([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083
#
#     gbasis = make_contractions(basis_dict, atsyms, atcoords)
#     gbasis_olp = overlap_integral(gbasis, coord_type="cartesian")
#
#     cbasis = CBasis(gbasis, atsyms, atcoords, coord_type="cartesian")
#     cbasis_olp = cbasis.olp()
#
#     assert cbasis_olp.shape[0] == cbasis_olp.shape[1] == \
#            gbasis_olp.shape[0] == gbasis_olp.shape[1] == \
#            cbasis.nbas
#
#     print(cbasis_olp)
#
#     assert np.allclose(np.diag(cbasis_olp), 1)
#     assert np.allclose(cbasis_olp, gbasis_olp)


# def test_overlap_horton_anorcc_bec():
#     """Test gbasis.integrals.libcint.CBasis.olp against GBasis's overlap matrix.
#
#     The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.
#
#     """
#     basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
#
#     basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]))
#     overlap_obj = Overlap(basis)
#     assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)
#
#     basis = make_contractions(basis_dict, ["Xe"], np.array([[0, 0, 0]]))
#     overlap_obj = Overlap(basis)
#     assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)
#
#
# def test_overlap_cartesian_norm_sto6g():
#     """Test the norm of gbasis.integrals.overlap_cartesian on the STO-6G basis set.
#
#     The contraction coefficients in STO-6G is such that the Cartesian contractions are not
#     normalized to past 3rd decimal places.
#
#     """
#     basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
#
#     basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
#     overlap_obj = Overlap(basis)
#     assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


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
    assert np.allclose(CBasis(basis, coord_type="cartesian").olp(), horton_overlap)


def test_overlap_horton_anorcc_bec():
    """Test gbasis.integrals.overlap.overlap_cartesian against HORTON's overlap matrix.

    The test case is diatomic with Be and C separated by 1.0 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    basis = make_contractions(
        basis_dict, ["Be", "C"], np.array([[0, 0, 0], [1.0 * 1.0 / 0.5291772083, 0, 0]])
    )
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, icenter=i.icenter,
                                charge=i.charge) for i in basis]

    horton_overlap = np.load(find_datafile("data_horton_bec_cart_overlap.npy"))
    assert np.allclose(CBasis(basis, coord_type="cartesian").olp(), horton_overlap)
