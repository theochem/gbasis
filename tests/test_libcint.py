"""Test gbasis.integrals.libcint."""

import pytest

import numpy as np
import numpy.testing as npt

from gbasis.integrals.angular_momentum import angular_momentum_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.moment import moment_integral
from gbasis.integrals.momentum import momentum_integral
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.point_charge import point_charge_integral

from gbasis.integrals.libcint import ELEMENTS, LIBCINT, CBasis

from gbasis.parsers import make_contractions, parse_nwchem

from utils import find_datafile


TEST_BASIS_SETS = [
    pytest.param("data_sto6g.nwchem", id="STO-6G"),
    pytest.param("data_631g.nwchem", id="6-31G"),
    pytest.param("data_ccpvdz.nwchem", id="cc-pVDZ"),
    # Slow tests:
    # pytest.param("data_ugbs.nwchem",   id="UGBS"),
    # pytest.param("data_anorcc.nwchem", id="ANO-RCC"),
]


TEST_SYSTEMS = [
    pytest.param(["He"], np.asarray([[0.0, 0.0, 0.0]]), id="He"),
    pytest.param(["C"], np.asarray([[0.0, 0.0, 0.0]]), id="C"),
    pytest.param(["H", "He"], np.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]), id="H,He"),
    pytest.param(["Be", "C"], np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), id="Be,C"),
    pytest.param(["H", "He", "Li"], np.eye(3, dtype=float), id="H,He,Li"),
]


TEST_COORD_TYPES = [
    pytest.param("cartesian", id="Cartesian"),
    pytest.param("spherical", id="Spherical"),
]


TEST_INTEGRALS = [
    pytest.param("overlap", id="Overlap"),
    pytest.param("kinetic_energy", id="KineticEnergy"),
    pytest.param("nuclear_attraction", id="NuclearAttraction"),
    pytest.param("momentum", id="Momentum"),
    pytest.param("angular_momentum", id="AngularMomentum"),
    pytest.param("electron_repulsion", id="ElectronRepulsion"),
    pytest.param("point_charge", id="PointCharge"),
    pytest.param("moment", id="Moment"),
]


@pytest.mark.parametrize("integral", TEST_INTEGRALS)
@pytest.mark.parametrize("coord_type", TEST_COORD_TYPES)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)
def test_integral(basis, atsyms, atcoords, coord_type, integral):
    r"""
    Test gbasis.integrals.libcint.CBasis integrals
    against the GBasis Python integrals.

    """
    atol, rtol = 1e-4, 1e-4

    atcoords = atcoords / 0.5291772083

    atnums = np.asarray([ELEMENTS.index(i) for i in atsyms], dtype=float)

    basis_dict = parse_nwchem(find_datafile(basis))

    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types=coord_type)

    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type=coord_type)

    if integral == "overlap":
        py_int = overlap_integral(py_basis)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.overlap()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "kinetic_energy":
        py_int = kinetic_energy_integral(py_basis)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.kinetic_energy()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "nuclear_attraction":
        py_int = nuclear_electron_attraction_integral(py_basis, atcoords, atnums)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.nuclear_attraction()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "angular_momentum":
        # py_int = angular_momentum_integral(py_basis)
        # npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        with pytest.raises(NotImplementedError):
            lc_int = lc_basis.angular_momentum(origin=np.zeros(3))
        # npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        return

    elif integral == "momentum":
        py_int = momentum_integral(py_basis)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        lc_int = lc_basis.momentum(origin=np.zeros(3))
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))

    elif integral == "electron_repulsion":
        py_int = electron_repulsion_integral(py_basis)
        npt.assert_array_equal(
            py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )
        lc_int = lc_basis.electron_repulsion()
        npt.assert_array_equal(
            lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )

    elif integral == "point_charge":
        charge_coords = np.asarray([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0], [-1.0, 2.0, -3.0]])
        charges = np.asarray([1.0, 0.666, -3.1415926])
        for i in range(1, len(charges) + 1):
            py_int = point_charge_integral(py_basis, charge_coords[:i], charges[:i])
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))
            lc_int = lc_basis.point_charge(charge_coords[:i], charges[:i])
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))

    elif integral == "moment":
        origin = np.zeros(3)
        orders = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ]
        )
        py_int = moment_integral(py_basis, origin, orders)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
        lc_int = lc_basis.moment(orders, origin=origin)
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))

    else:
        raise ValueError("Invalid integral name '{integral}' passed")

    npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
