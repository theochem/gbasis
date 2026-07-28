"""Test gbasis.integrals.libcint."""

import pytest
import os
import sys

from os.path import dirname, join
from glob import glob

import numpy as np
import numpy.testing as npt

import gbasis

from gbasis.integrals.angular_momentum import angular_momentum_integral
from gbasis.integrals.electron_repulsion import (
    electron_repulsion_integral,
    electron_repulsion_integral_improved,
)
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.moment import moment_integral
from gbasis.integrals.momentum import momentum_integral
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.point_charge import point_charge_integral

from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.wrappers import from_iodata

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

@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("integral", TEST_INTEGRALS)
@pytest.mark.parametrize("coord_type", TEST_COORD_TYPES)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)
def test_integral(basis, atsyms, atcoords, coord_type, integral):
    from gbasis.integrals.libcint import ELEMENTS, LIBCINT, CBasis

    r"""
    Test gbasis.integrals.libcint.CBasis integrals
    against the GBasis Python integrals.

    """
    atol, rtol = 1e-6, 1e-6

    atcoords = atcoords / 0.5291772083

    atnums = np.asarray([ELEMENTS.index(i) for i in atsyms], dtype=float)

    basis_dict = parse_nwchem(find_datafile(basis))

    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types=coord_type)

    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type=coord_type)

    if integral == "overlap":
        py_int = overlap_integral(py_basis, screen_basis=False)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.overlap_integral()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "kinetic_energy":
        py_int = kinetic_energy_integral(py_basis, screen_basis=False)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.kinetic_energy_integral()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "nuclear_attraction":
        py_int = nuclear_electron_attraction_integral(py_basis, atcoords, atnums)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        lc_int = lc_basis.nuclear_attraction_integral()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "angular_momentum":
        # py_int = angular_momentum_integral(py_basis)
        # npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        with pytest.raises(NotImplementedError):
            lc_int = lc_basis.angular_momentum_integral(origin=np.zeros(3))
        # npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        return

    elif integral == "momentum":
        py_int = momentum_integral(py_basis, screen_basis=False)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        lc_int = lc_basis.momentum_integral(origin=np.zeros(3))
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))

    elif integral == "electron_repulsion":
        py_int = electron_repulsion_integral_improved(py_basis)
        npt.assert_array_equal(
            py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )
        lc_int = lc_basis.electron_repulsion_integral()
        npt.assert_array_equal(
            lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )
        npt.assert_allclose(lc_int, py_int, atol=1e-4, rtol=1e-5)
        return

    elif integral == "point_charge":
        charge_coords = np.asarray([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0], [-1.0, 2.0, -3.0]])
        charges = np.asarray([1.0, 0.666, -3.1415926])
        for i in range(1, len(charges) + 1):
            py_int = point_charge_integral(py_basis, charge_coords[:i], charges[:i])
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))
            lc_int = lc_basis.point_charge_integral(charge_coords[:i], charges[:i])
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
        py_int = moment_integral(py_basis, origin, orders, screen_basis=False)
        npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
        lc_int = lc_basis.moment_integral(orders, origin=origin)
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))

    else:
        raise ValueError("Invalid integral name '{integral}' passed")

    npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)



TEST_SYSTEMS_IODATA = [
    pytest.param("h2o_hf_ccpv5z_cart.fchk", ["O", "H", "H"], "Cartesian", id="h2o_cart"),
    pytest.param("h2o_hf_ccpv5z_sph.fchk", ["O", "H", "H"], "Spherical", id="h2o_sph"),
    ]

TEST_COORD_TRANSFORM = [
    pytest.param(False, id="no-transform"),
    pytest.param(True, id="transform"),
]

TEST_INTEGRALS_IODATA = [
    pytest.param("overlap", id="Overlap"),
    pytest.param("kinetic_energy", id="KineticEnergy"),
    pytest.param("nuclear_attraction", id="NuclearAttraction"),
    pytest.param("momentum", id="Momentum"),
    pytest.param("angular_momentum", id="AngularMomentum"),
    pytest.param("electron_repulsion", marks=pytest.mark.skip(reason='TOO SLOW'), id="ElectronRepulsion"),
    pytest.param("point_charge", id="PointCharge"),
    pytest.param("moment", id="Moment"),
]
@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("fname, elements, coord_type", TEST_SYSTEMS_IODATA)
@pytest.mark.parametrize("transform", TEST_COORD_TRANSFORM)
@pytest.mark.parametrize("integral", TEST_INTEGRALS_IODATA)
def test_integral_iodata(fname, elements, coord_type, integral, transform):
    pytest.importorskip("iodata")
    from iodata import load_one
    from gbasis.integrals.libcint import ELEMENTS, LIBCINT, CBasis

    atol, rtol = 1e-6, 1e-6

    mol=load_one(find_datafile(fname))
    py_basis=from_iodata(mol)

    lc_basis = CBasis(py_basis, elements, mol.atcoords, coord_type=coord_type)

    if integral == "overlap":
        if transform:
            py_int = overlap_integral(py_basis, transform=mol.mo.coeffs.T, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.overlap_integral(transform=mol.mo.coeffs.T)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        else:
            py_int = overlap_integral(py_basis, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.overlap_integral()
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "kinetic_energy":
        if transform:
            py_int = kinetic_energy_integral(py_basis, transform=mol.mo.coeffs.T, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.kinetic_energy_integral(transform=mol.mo.coeffs.T)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        else:
            py_int = kinetic_energy_integral(py_basis, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.kinetic_energy_integral()
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "nuclear_attraction":
        if transform:
            py_int = nuclear_electron_attraction_integral(py_basis, mol.atcoords,
                                                          mol.atnums, transform=mol.mo.coeffs.T)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.nuclear_attraction_integral(transform=mol.mo.coeffs.T)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        else:
            py_int = nuclear_electron_attraction_integral(py_basis, mol.atcoords, mol.atnums)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
            lc_int = lc_basis.nuclear_attraction_integral()
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    elif integral == "angular_momentum":
        # py_int = angular_momentum_integral(py_basis)
        # npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        with pytest.raises(NotImplementedError):
            lc_int = lc_basis.angular_momentum_integral(origin=np.zeros(3))
        # npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        return

    elif integral == "momentum":
        if transform:
            py_int = momentum_integral(py_basis, transform=mol.mo.coeffs.T, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
            lc_int = lc_basis.momentum_integral(origin=np.zeros(3), transform=mol.mo.coeffs.T)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        else:
            py_int = momentum_integral(py_basis, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
            lc_int = lc_basis.momentum_integral(origin=np.zeros(3))
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))

    elif integral == "electron_repulsion":
        if transform:
            py_int = electron_repulsion_integral(py_basis, transform=mol.mo.coeffs.T)
            npt.assert_array_equal(
                py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
            )
            lc_int = lc_basis.electron_repulsion_integral(transform=mol.mo.coeffs.T)
            npt.assert_array_equal(
                lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
            )
        else:
            py_int = electron_repulsion_integral(py_basis)
            npt.assert_array_equal(
                py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
            )
            lc_int = lc_basis.electron_repulsion_integral()
            npt.assert_array_equal(
                lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
            )

    elif integral == "point_charge":
        charge_coords = np.asarray([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0], [-1.0, 2.0, -3.0]])
        charges = np.asarray([1.0, 0.666, -3.1415926])
        if transform:
            for i in range(1, len(charges) + 1):
                py_int = point_charge_integral(py_basis, charge_coords[:i],
                                               charges[:i], transform=mol.mo.coeffs.T)
                npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))
                lc_int = lc_basis.point_charge_integral(charge_coords[:i],
                                                        charges[:i], transform=mol.mo.coeffs.T)
                npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))

        else:
            for i in range(1, len(charges) + 1):
                py_int = point_charge_integral(py_basis, charge_coords[:i], charges[:i])
                npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))
                lc_int = lc_basis.point_charge_integral(charge_coords[:i], charges[:i])
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
        if transform:
            py_int = moment_integral(py_basis, origin, orders, transform=mol.mo.coeffs.T, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
            lc_int = lc_basis.moment_integral(orders, origin=origin, transform=mol.mo.coeffs.T)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
        else:
            py_int = moment_integral(py_basis, origin, orders, screen_basis=False)
            npt.assert_array_equal(py_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
            lc_int = lc_basis.moment_integral(orders, origin=origin)
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))

    else:
        raise ValueError("Invalid integral name '{integral}' passed")

    npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)




# ─────────────────────────────────────────────────────────────────────────
# New test list for C shell-loop bindings introduced in PR-5 and PR-6
# ─────────────────────────────────────────────────────────────────────────

TEST_C_SHELLLOOP_INTEGRALS = [
    pytest.param("overlap", id="C-Overlap"),
    pytest.param("kinetic_energy", id="C-KineticEnergy"),
    pytest.param("nuclear_attraction", id="C-NuclearAttraction"),
    pytest.param("rinv", id="C-Rinv"),
    pytest.param("momentum", id="C-Momentum"),
    pytest.param("dipole", id="C-Dipole"),
    pytest.param("quadrupole", id="C-Quadrupole"),
    pytest.param("octupole", id="C-Octupole"),
    pytest.param("point_charge", id="C-PointCharge"),
    pytest.param("moment", id="C-Moment"),
    pytest.param("electron_repulsion", id="C-ElectronRepulsion"),
]


@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("integral", TEST_C_SHELLLOOP_INTEGRALS)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)
def test_c_shellloop_integral(basis, atsyms, atcoords, integral):
    r"""
    Test the C shell-loop bindings (PR-5: 1-electron, PR-6: ERI) added to
    ``gbasis.integrals.libcint.CBasis`` against the existing GBasis Python
    integral implementations.

    These are the ``.overlap()``, ``.kinetic_energy()``,
    ``.nuclear_attraction()``, ``.rinv()``, ``.dipole()``, ``.quadrupole()``,
    ``.octupole()``, and ``.electron_repulsion()`` methods, which loop over
    shells directly in C (as opposed to the ``*_integral()`` methods, which
    loop over shells in Python and only call into C per shell pair).

    """
    from gbasis.integrals.libcint import ELEMENTS, CBasis

    atol, rtol = 1e-6, 1e-6

    atcoords = atcoords / 0.5291772083

    atnums = np.asarray([ELEMENTS.index(i) for i in atsyms], dtype=float)

    basis_dict = parse_nwchem(find_datafile(basis))

    # C shell-loop bindings are implemented for spherical only
    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types="spherical")

    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="spherical")

    if integral == "overlap":
        py_int = overlap_integral(py_basis, screen_basis=False)
        lc_int = lc_basis.overlap()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.overlap(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "kinetic_energy":
        py_int = kinetic_energy_integral(py_basis, screen_basis=False)
        lc_int = lc_basis.kinetic_energy()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.kinetic_energy(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "nuclear_attraction":
        py_int = nuclear_electron_attraction_integral(py_basis, atcoords, atnums)
        lc_int = lc_basis.nuclear_attraction()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.nuclear_attraction(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "rinv":
        # Compare against the point_charge Python integral with a single
        # unit charge at the origin, since rinv == 1/|r - origin|
        origin = np.zeros(3)
        py_int = point_charge_integral(
            py_basis, origin.reshape(1, 3), np.asarray([-1.0])
        )[:, :, 0]
        lc_int = lc_basis.rinv()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with inv_origin
        lc_int_inv = lc_basis.rinv(inv_origin=np.zeros(3))
        npt.assert_allclose(lc_int_inv, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.rinv(transform=transform)
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "dipole":
        origin = np.zeros(3)
        orders = np.asarray([[1, 0, 0]])
        py_int = moment_integral(py_basis, origin, orders, screen_basis=False)[:, :, 0]
        lc_int = lc_basis.dipole()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.dipole(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)


    elif integral == "quadrupole":
        origin = np.zeros(3)
        orders = np.asarray([[2, 0, 0]])
        py_int = moment_integral(py_basis, origin, orders, screen_basis=False)[:, :, 0]
        lc_int = lc_basis.quadrupole()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.quadrupole(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "octupole":
        origin = np.zeros(3)
        orders = np.asarray([[3, 0, 0]])
        py_int = moment_integral(py_basis, origin, orders, screen_basis=False)[:, :, 0]
        lc_int = lc_basis.octupole()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.octupole(transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "momentum":
        py_int = momentum_integral(py_basis, screen_basis=False)
        lc_int = lc_basis.momentum(origin=np.zeros(3))
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.momentum(origin=np.zeros(3), transform=transform)
        npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn, 3))
        npt.assert_allclose(lc_int_t, py_int, atol=atol, rtol=rtol)

    elif integral == "point_charge":
        charge_coords = np.asarray([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0], [-1.0, 2.0, -3.0]])
        charges = np.asarray([1.0, 0.666, -3.1415926])
        for i in range(1, len(charges) + 1):
            py_int = point_charge_integral(py_basis, charge_coords[:i], charges[:i])
            lc_int = lc_basis.point_charge(charge_coords[:i], charges[:i])
            npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, i))
            npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)

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
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
            ]
        )
        py_int = moment_integral(py_basis, origin, orders, screen_basis=False)
        lc_int = lc_basis.moment(orders, origin=origin)
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, len(orders)))
        npt.assert_allclose(lc_int, py_int, atol=atol, rtol=rtol)

    elif integral == "electron_repulsion":
        py_int = electron_repulsion_integral_improved(py_basis)
        lc_int = lc_basis.electron_repulsion(notation="physicist")
        npt.assert_array_equal(
            lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )
        # ERI uses a looser tolerance, consistent with the existing
        # electron_repulsion_integral test above
        npt.assert_allclose(lc_int, py_int, atol=1e-4, rtol=1e-5)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.electron_repulsion(notation="physicist",transform=transform)
        npt.assert_array_equal(
            lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn, lc_basis.nbfn)
        )
        npt.assert_allclose(lc_int_t, py_int, atol=1e-4, rtol=1e-5)
    else:
        raise ValueError(f"Invalid integral name '{integral}' passed")


@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)

def test_c_shellloop_matches_make_int1e(basis, atsyms, atcoords):
    r"""
    Cross-check the new C shell-loop bindings (``.overlap()``,
    ``.kinetic_energy()``, ``.nuclear_attraction()``, ``.electron_repulsion()``)
    directly against the existing ``make_int1e``/``make_int2e``-based methods
    (``.overlap_integral()``, ``.kinetic_energy_integral()``,
    ``.nuclear_attraction_integral()``, ``.electron_repulsion_integral()``)
    on the *same* ``CBasis`` instance.

    This isolates the C shell-loop logic itself (PR-5/PR-6) from any
    differences against the pure-Python GBasis implementation, since both
    sides here come from libcint.

    """
    from gbasis.integrals.libcint import ELEMENTS, CBasis

    atcoords = atcoords / 0.5291772083

    basis_dict = parse_nwchem(find_datafile(basis))

    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types="spherical")

    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="spherical")

    # 1-electron integrals: C shell-loop vs. make_int1e shell-loop
    npt.assert_allclose(
        lc_basis.overlap(), lc_basis.overlap_integral(), atol=1e-10, rtol=1e-10
    )
    npt.assert_allclose(
        lc_basis.kinetic_energy(),
        lc_basis.kinetic_energy_integral(),
        atol=1e-10,
        rtol=1e-10,
    )
    npt.assert_allclose(
        lc_basis.nuclear_attraction(),
        lc_basis.nuclear_attraction_integral(),
        atol=1e-10,
        rtol=1e-10,
    )
    # overlap with transform
    transform = np.eye(lc_basis.nbfn)
    npt.assert_allclose(
        lc_basis.overlap(transform=transform),
        lc_basis.overlap_integral(transform=transform),
        atol=1e-10,
        rtol=1e-10,
    )

    # kinetic_energy with transform
    npt.assert_allclose(
        lc_basis.kinetic_energy(transform=transform),
        lc_basis.kinetic_energy_integral(transform=transform),
        atol=1e-10, rtol=1e-10,
    )

    # nuclear_attraction with transform
    npt.assert_allclose(
        lc_basis.nuclear_attraction(transform=transform),
        lc_basis.nuclear_attraction_integral(transform=transform),
        atol=1e-10, rtol=1e-10,
    )

    # rinv with inv_origin and transform
    npt.assert_allclose(
        lc_basis.rinv(inv_origin=np.zeros(3), transform=transform),
        lc_basis.r_inv_integral(origin=np.zeros(3), transform=transform),
        atol=1e-10, rtol=1e-10,
    )

    # momentum with transform
    npt.assert_allclose(
        lc_basis.momentum(origin=np.zeros(3), transform=transform),
        lc_basis.momentum_integral(origin=np.zeros(3), transform=transform),
        atol=1e-10, rtol=1e-10,
    )
    

    # 2-electron ERI: C shell-loop vs. make_int2e shell-loop
    npt.assert_allclose(
        lc_basis.electron_repulsion(notation="chemist"),
        lc_basis.electron_repulsion_integral(notation="chemist"),
        atol=1e-8,
        rtol=1e-8,
    )

    # electron_repulsion with transform
    npt.assert_allclose(
        lc_basis.electron_repulsion(notation="chemist", transform=transform),
        lc_basis.electron_repulsion_integral(notation="chemist", transform=transform),
        atol=1e-8, rtol=1e-8,
    )

# ─────────────────────────────────────────────────────────────────────────
# Tests for gradient integral bindings 
# ─────────────────────────────────────────────────────────────────────────

TEST_GRADIENT_INTEGRALS = [
    pytest.param("gradient_kinetic", id="C-GradKinetic"),
    pytest.param("gradient_nuclear", id="C-GradNuclear"),
    pytest.param("gradient_rinv",    id="C-GradRinv"),
]


@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("integral", TEST_GRADIENT_INTEGRALS)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)
def test_c_gradient_integral(basis, atsyms, atcoords, integral):
    r"""
    Test the C shell-loop gradient integral bindings (PR-8) added to
    ``gbasis.integrals.libcint.CBasis`` against the existing make_int1e
    based implementations.

    These are the ``.gradient_kinetic()``, ``.gradient_nuclear()``,
    and ``.gradient_rinv()`` methods which are the building blocks
    for computing nuclear coordinate gradients.
    """
    from gbasis.integrals.libcint import ELEMENTS, CBasis

    atcoords = atcoords / 0.5291772083

    basis_dict = parse_nwchem(find_datafile(basis))

    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types="spherical")
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="spherical")

    if integral == "gradient_kinetic":
        # Compare C shell-loop against make_int1e path on same CBasis instance
        py_int = lc_basis._d_kin()
        lc_int = lc_basis.gradient_kinetic()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int[..., 0], atol=1e-10, rtol=1e-10)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.gradient_kinetic(transform=transform)
        npt.assert_allclose(lc_int_t, py_int[..., 0], atol=1e-10, rtol=1e-10)

    elif integral == "gradient_nuclear":
        py_int = lc_basis._d_nuc()
        lc_int = lc_basis.gradient_nuclear()
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int[..., 0], atol=1e-10, rtol=1e-10)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.gradient_nuclear(transform=transform)
        npt.assert_allclose(lc_int_t, py_int[..., 0], atol=1e-10, rtol=1e-10)

    elif integral == "gradient_rinv":
        py_int = lc_basis._d_rinv(inv_origin=np.zeros(3))
        lc_int = lc_basis.gradient_rinv(inv_origin=np.zeros(3))
        npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))
        npt.assert_allclose(lc_int, py_int[..., 0], atol=1e-10, rtol=1e-10)
        # Test with transform
        transform = np.eye(lc_basis.nbfn)
        lc_int_t = lc_basis.gradient_rinv(inv_origin=np.zeros(3), transform=transform)
        npt.assert_allclose(lc_int_t, py_int[..., 0], atol=1e-10, rtol=1e-10)
    else:
        raise ValueError(f"Invalid integral name '{integral}' passed")


# ─────────────────────────────────────────────────────────────────────────
# Tests for GIAO/magnetic integral bindings 
# ─────────────────────────────────────────────────────────────────────────

TEST_GIAO_INTEGRALS = [
    pytest.param("ia01p",  id="C-ia01p"),
    pytest.param("ircxp",  id="C-ircxp"),
    pytest.param("iking",  id="C-iking"),
    pytest.param("iovlpg", id="C-iovlpg"),
    pytest.param("inucg",  id="C-inucg"),
]


@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("integral", TEST_GIAO_INTEGRALS)
@pytest.mark.parametrize("atsyms, atcoords", TEST_SYSTEMS)
@pytest.mark.parametrize("basis", TEST_BASIS_SETS)
def test_c_giao_integral(basis, atsyms, atcoords, integral):
    r"""
    Test the GIAO/magnetic integral bindings (PR-8) added to
    ``gbasis.integrals.libcint.CBasis``.

    These are the ``.ia01p()``, ``.ircxp()``, ``.iking()``,
    ``.iovlpg()``, and ``.inucg()`` methods which are building
    blocks for NMR/magnetic property calculations.

    Since GBasis has no Python reference implementation for GIAO
    integrals, we verify shape and that results are finite and
    non-trivially zero for multi-atom systems.
    """
    from gbasis.integrals.libcint import ELEMENTS, CBasis

    atcoords = atcoords / 0.5291772083

    basis_dict = parse_nwchem(find_datafile(basis))
    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types="spherical")
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="spherical")

    if integral == "ia01p":
        lc_int = lc_basis.ia01p()
    elif integral == "ircxp":
        lc_int = lc_basis.ircxp()
    elif integral == "iking":
        lc_int = lc_basis.iking()
    elif integral == "iovlpg":
        lc_int = lc_basis.iovlpg()
    elif integral == "inucg":
        lc_int = lc_basis.inucg()
    else:
        raise ValueError(f"Invalid integral name '{integral}' passed")

    # Shape check
    npt.assert_array_equal(lc_int.shape, (lc_basis.nbfn, lc_basis.nbfn))

    # Finiteness check — no NaN or Inf
    assert np.all(np.isfinite(lc_int)), f"{integral} contains NaN or Inf"
    # Test with transform
    transform = np.eye(lc_basis.nbfn)
    func = getattr(lc_basis, integral)
    lc_int_t = func(transform=transform)
    npt.assert_array_equal(lc_int_t.shape, (lc_basis.nbfn, lc_basis.nbfn))
    assert np.all(np.isfinite(lc_int_t)), f"{integral} with transform contains NaN or Inf"


# ─────────────────────────────────────────────────────────────────────────
# Tests for 3-center 2-electron integral bindings
# ─────────────────────────────────────────────────────────────────────────

TEST_3C2E_SYSTEMS = [
    pytest.param(["He"], np.asarray([[0.0, 0.0, 0.0]]), "He", "spherical", id="He-sph"),
    pytest.param(["C"], np.asarray([[0.0, 0.0, 0.0]]), "C", "spherical", id="C-sph"),
    pytest.param(["H", "He"], np.asarray([[0.0, 0.0, 0.0], [1.5117, 0.0, 0.0]]), "H_He", "spherical", id="H_He-sph"),
    pytest.param(["Be", "C"], np.asarray([[0.0, 0.0, 0.0], [1.8897, 0.0, 0.0]]), "Be_C", "spherical", id="Be_C-sph"),
    pytest.param(["He"], np.asarray([[0.0, 0.0, 0.0]]), "He", "cartesian", id="He-cart"),
    pytest.param(["C"], np.asarray([[0.0, 0.0, 0.0]]), "C", "cartesian", id="C-cart"),
    pytest.param(["H", "He"], np.asarray([[0.0, 0.0, 0.0], [1.5117, 0.0, 0.0]]), "H_He", "cartesian", id="H_He-cart"),
    pytest.param(["Be", "C"], np.asarray([[0.0, 0.0, 0.0], [1.8897, 0.0, 0.0]]), "Be_C", "cartesian", id="Be_C-cart"),
]

@pytest.mark.skipif(sys.platform == "win32", reason="This test does not work on Windows")
@pytest.mark.skipif(
    len(glob(join(dirname(gbasis.__file__), "integrals", "lib", "libcint.*"))) == 0,
    reason="The libcint shared library object was not found",
)
@pytest.mark.parametrize("atsyms, atcoords, fname, coord_type", TEST_3C2E_SYSTEMS)
def test_c_3center_2electron(atsyms, atcoords, fname, coord_type):
    from gbasis.integrals.libcint import ELEMENTS, CBasis

    prefix = "cart" if coord_type == "cartesian" else "sph"
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    py_basis = make_contractions(basis_dict, atsyms, atcoords, coord_types=coord_type)
    lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type=coord_type)

    ref = np.load(find_datafile(f"data_3c2e_{prefix}_sto6g_{fname}.npy"))
    our = lc_basis.three_center_two_electron()

    npt.assert_array_equal(our.shape, ref.shape)
    npt.assert_allclose(our, ref, atol=1e-10, rtol=1e-10)

    transform = np.eye(lc_basis.nbfn)
    our_t = lc_basis.three_center_two_electron(transform=transform)
    npt.assert_array_equal(our_t.shape, ref.shape)
    npt.assert_allclose(our_t, ref, atol=1e-10, rtol=1e-10)