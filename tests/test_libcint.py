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
#     atsyms = ["Be", "C"]
#
#     atcoords = np.array([[0., 0., 0.], [1.0, 0., 0.]]) / 0.5291772083
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
