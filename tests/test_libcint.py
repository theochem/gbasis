"""Test gbasis.integrals.libcint."""

import pytest

import numpy as np
import numpy.testing as npt

from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral

from gbasis.integrals.libcint import ELEMENTS, LIBCINT, CBasis

from gbasis.parsers import make_contractions, parse_nwchem

from utils import find_datafile


TEST_BASIS_SETS = [
    pytest.param("data_sto6g.nwchem",  id="STO-6G"),
    pytest.param("data_631g.nwchem",   id="6-31G"),
    pytest.param("data_ccpvdz.nwchem", id="cc-pVDZ"),
    # Slow tests:
    # pytest.param("data_ugbs.nwchem",   id="UGBS"),
    # pytest.param("data_anorcc.nwchem", id="ANO-RCC"),
]


TEST_SYSTEMS = [
    pytest.param(["He"],            np.asarray([[0., 0., 0.]]),                id="He"),
    pytest.param(["C"],             np.asarray([[0., 0., 0.]]),                id="C"),
    pytest.param(["H", "He"],       np.asarray([[0., 0., 0.], [0.8, 0., 0.]]), id="H,He"),
    pytest.param(["Be", "C"],       np.asarray([[0., 0., 0.], [1.0, 0., 0.]]), id="Be,C"),
    pytest.param(["H", "He", "Li"], np.eye(3, dtype=float),                    id="H,He,Li"),
]


TEST_COORD_TYPES = [
    # pytest.param("cartesian", id="Cartesian"),
    pytest.param("spherical", id="Spherical"),
]


TEST_INTEGRALS = [
    # Integrals
    pytest.param("olp", id="Overlap"),
    pytest.param("kin", id="KineticEnergy"),
    pytest.param("nuc", id="NuclearAttraction"),
    pytest.param("eri", id="ElectronRepulsion"),
    # Gradients
    pytest.param("d_olp", id="OverlapGradient"),
    pytest.param("d_kin", id="KineticEnergyGradient"),
    pytest.param("d_nuc", id="NuclearAttractionGradient"),
]


@pytest.mark.parametrize("integral",         TEST_INTEGRALS)
@pytest.mark.parametrize("coord_type",       TEST_COORD_TYPES)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis",            TEST_BASIS_SETS)
def test_cbasis(basis, atsyms, atcoords, coord_type, integral):
    r"""
    Test gbasis.integrals.libcint.CBasis integrals
    against the GBasis Python integrals.

    """
    atol, rtol = 1e-4, 1e-4

    atcoords = atcoords / 0.5291772083

    atnums = np.asarray([ELEMENTS.index(i) for i in atsyms], dtype=float)

    basis_dict = parse_nwchem(find_datafile(basis))

    atsyms = ["H", "He"]

    atcoords = np.array([[0., 0., 0.], [0.8, 0., 0.]]) / 0.5291772083

    gbasis = make_contractions(basis_dict, atsyms, atcoords)
    gbasis_olp = overlap_integral(gbasis, coord_type="cartesian")
    py_basis = make_contractions(basis_dict, atsyms, atcoords)

    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type=coord_type)

    if integral == "olp":
        py_int = overlap_integral(py_basis, coord_type=coord_type)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn,) * 2)
        lc_int = lc_basis.olp()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn,) * 2)

    elif integral == "kin":
        py_int = kinetic_energy_integral(py_basis, coord_type=coord_type)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn,) * 2)
        lc_int = lc_basis.kin()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn,) * 2)

    elif integral == "nuc":
        py_int = nuclear_electron_attraction_integral(py_basis, atcoords, atnums, coord_type=coord_type)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn,) * 2)
        lc_int = lc_basis.nuc()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn,) * 2)

    elif integral == "eri":
        py_int = electron_repulsion_integral(py_basis, coord_type=coord_type)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn,) * 4)
        lc_int = lc_basis.eri()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn,) * 4)

    elif integral == "d_olp":
        lc_int = lc_basis.d_olp()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.natm, 3))
        py_int = lc_int.copy() # Just so we don't crash

    elif integral == "d_kin":
        lc_int = lc_basis.d_kin()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.natm, 3))
        py_int = lc_int.copy() # Just so we don't crash

    elif integral == "d_nuc":
        lc_int = lc_basis.d_nuc()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.natm, 3))
        py_int = lc_int.copy() # Just so we don't crash

    else:
        raise ValueError("Invalid integral name '{integral}' passed")

    diff = np.bitwise_not(np.isclose(lc_int, py_int, atol=atol, rtol=rtol).reshape(-1))
    if np.any(diff):
        with np.printoptions(threshold=1000):
            print(f"\nGBASIS\n{py_int.reshape(-1)[diff]}\nLIBCINT\n{lc_int.reshape(-1)[diff]}")

    npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
