"""Test gbasis.integrals.libcint."""

import pytest

import numpy as np
import numpy.testing as npt

from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral

from gbasis.integrals.libcint import ELEMENTS, CBasis

from gbasis.parsers import make_contractions, parse_nwchem

from utils import find_datafile


CASES = [
    ("UGBS", "H", np.asarray([[0., 0., 0.]]) / 0.5291772083),
    ("UGBS", "He", np.asarray([[0., 0., 0.]]) / 0.5291772083),
    # ("UGBS", "H,He", np.asarray([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083),
    # ("UGBS", "Be,C", np.asarray([[0., 0., 0.], [1., 0., 0.]]) / 0.5291772083),
    # ("ANORCC", "H,He", np.asarray([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083),
    # ("ANORCC", "Be,C", np.asarray([[0., 0., 0.], [1., 0., 0.]]) / 0.5291772083),
]
r"""
Test systems for gbasis.integrals.libcint.

"""


@pytest.mark.parametrize("basis, atsyms, atcoords", CASES)
def test_cbasis_overlap(basis, atsyms, atcoords):
    r"""
    Test gbasis.integrals.libcint.CBasis overlap integrals
    against the GBasis Python overlap integrals.

    """
    atsyms = [sym.strip() for sym in atsyms.split(",")]
    basis_dict = parse_nwchem(find_datafile(f"data_{basis.lower()}.nwchem"))

    py_basis = make_contractions(basis_dict, atsyms, atcoords)
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="cartesian")

    py_olp = overlap_integral(py_basis, coord_type="cartesian")
    lc_olp = lc_basis.olp()

    npt.assert_array_equal(py_olp.shape, lc_basis.nbas)
    npt.assert_array_equal(lc_olp.shape, lc_basis.nbas)
    npt.assert_allclose(np.diag(py_olp), 1)
    npt.assert_allclose(lc_olp, py_olp)


@pytest.mark.parametrize("basis, atsyms, atcoords", CASES)
def test_cbasis_kinetic(basis, atsyms, atcoords):
    r"""
    Test gbasis.integrals.libcint.CBasis kinetic energy integrals
    against the GBasis Python kinetic energy integrals.

    """
    atsyms = [sym.strip() for sym in atsyms.split(",")]
    basis_dict = parse_nwchem(find_datafile(f"data_{basis.lower()}.nwchem"))

    py_basis = make_contractions(basis_dict, atsyms, atcoords)
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="cartesian")

    py_kin = kinetic_energy_integral(py_basis, coord_type="cartesian")
    lc_kin = lc_basis.kin()

    npt.assert_array_equal(py_kin.shape, lc_basis.nbas)
    npt.assert_array_equal(lc_kin.shape, lc_basis.nbas)
    npt.assert_allclose(lc_kin, py_kin)


@pytest.mark.parametrize("basis, atsyms, atcoords", CASES)
def test_cbasis_nuclear(basis, atsyms, atcoords):
    r"""
    Test gbasis.integrals.libcint.CBasis nuclear electron attraction integrals
    against the GBasis Python nuclear electron attraction integrals.

    """
    atsyms = [sym.strip() for sym in atsyms.split(",")]
    basis_dict = parse_nwchem(find_datafile(f"data_{basis.lower()}.nwchem"))

    py_basis = make_contractions(basis_dict, atsyms, atcoords)
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="cartesian")

    atnums = np.asarray([ELEMENTS.index(i) for i in atsyms], dtype=float)
    py_nuc = nuclear_electron_attraction_integral(py_basis, atcoords, atnums, coord_type="cartesian")
    lc_nuc = lc_basis.nuc()

    npt.assert_array_equal(py_nuc.shape, lc_basis.nbas)
    npt.assert_array_equal(lc_nuc.shape, lc_basis.nbas)
    npt.assert_allclose(lc_nuc, py_nuc)


@pytest.mark.parametrize("basis, atsyms, atcoords", CASES)
def test_cbasis_eri(basis, atsyms, atcoords):
    r"""
    Test gbasis.integrals.libcint.CBasis electron repulsion integrals
    against the GBasis Python electron repulsion integrals.

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
    basis_dict = parse_nwchem(find_datafile("data_ugbs.nwchem"))

    atsyms = ["H", "He"]

    atcoords = np.array([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083

    gbasis = make_contractions(basis_dict, atsyms, atcoords)
    gbasis_olp = overlap_integral(gbasis, coord_type="cartesian")

    py_eri = electron_repulsion_integral(py_basis, coord_type="cartesian")
    lc_eri = lc_basis.eri()
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
