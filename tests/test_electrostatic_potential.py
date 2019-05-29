"""Tests for gbasis.electrostatic_potential."""
from gbasis.electrostatic_potential import (
    _electrostatic_potential_base,
    electrostatic_potential_cartesian,
    electrostatic_potential_mix,
    electrostatic_potential_spherical,
)
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_electrostatic_potential_base():
    """Test gbasis.electrostatic_potential._electorstatic_potential_base.

    Tested by using point_charge.point_charge_cartesian.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    # check density_matrix type
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103).tolist(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
        )
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103).flatten(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
        )
    # check nuclear_coords
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3).tolist(),
            np.array([1, 2]),
            "cartesian",
        )
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 4),
            np.array([1, 2]),
            "cartesian",
        )
    # check nuclear charges
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).reshape(1, 2),
            "cartesian",
        )
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).tolist(),
            "cartesian",
        )
    # check density_matrix symmetry
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.eye(103, 102),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
        )
    # check nuclear_coords and nuclear_charges shapes
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(3, 3),
            np.array([1, 2]),
            "cartesian",
        )
    # check threshold_dist types
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
            threshold_dist=None,
        )
    # check threshold_dist value
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
            threshold_dist=-0.1,
        )
    # check coord_types type
    with pytest.raises(TypeError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "bad",
        )
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.identity(88),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "cartesian",
        )
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            "spherical",
        )
    with pytest.raises(ValueError):
        _electrostatic_potential_base(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            ["spherical"] * 9,
        )


def test_electrostatic_potential_cartesian():
    """Test gbasis.electrostatic_potential.electorstatic_potential_cartesian.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        electrostatic_potential_cartesian(
            basis, np.identity(103), grid_3d, coords, np.array([1, 2])
        ),
        horton_nucattract,
    )


def test_electrostatic_potential_spherical():
    """Test gbasis.electrostatic_potential.electorstatic_potential_spherical.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        electrostatic_potential_spherical(
            basis, np.identity(88), grid_3d, coords, np.array([1, 2])
        ),
        horton_nucattract,
    )


def test_electrostatic_potential_mix():
    """Test gbasis.electrostatic_potential.electorstatic_potential_mix.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        electrostatic_potential_mix(
            basis, np.identity(88), grid_3d, coords, np.array([1, 2]), ["spherical"] * 9
        ),
        horton_nucattract,
    )
    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        electrostatic_potential_mix(
            basis, np.identity(103), grid_3d, coords, np.array([1, 2]), ["cartesian"] * 9
        ),
        horton_nucattract,
    )
