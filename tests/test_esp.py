"""Tests for gbasis.evals.esp."""
from gbasis.evals.esp import evaluate_electrostatic_potential
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_electrostatic_potential_raises():
    """Test gbasis.evals.esp.electorstatic_potential type errors."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    # check density_matrix type
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103).tolist(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103).flatten(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    # check nuclear_coords
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3).tolist(),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 4),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    # check nuclear charges
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).reshape(1, 2),
            coord_type="cartesian",
        )
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).tolist(),
            coord_type="cartesian",
        )
    # check density_matrix symmetry
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.eye(103, 102),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    # check nuclear_coords and nuclear_charges shapes
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(3, 3),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    # check threshold_dist types
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
            threshold_dist=None,
        )
    # check threshold_dist value
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
            threshold_dist=-0.1,
        )
    # check coord_types type
    with pytest.raises(TypeError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="bad",
        )
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(88),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="cartesian",
        )
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type="spherical",
        )
    with pytest.raises(ValueError):
        evaluate_electrostatic_potential(
            basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            coord_type=["spherical"] * 9,
        )


def test_electrostatic_potential_cartesian():
    """Test gbasis.evals.esp.electorstatic_potential for cartesian coordinate type.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        evaluate_electrostatic_potential(
            basis, np.identity(103), grid_3d, coords, np.array([1, 2]), coord_type="cartesian"
        ),
        horton_nucattract,
    )


def test_electrostatic_potential_spherical():
    """Test gbasis.evals.esp.electorstatic_potential for spherical coordinate type.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        evaluate_electrostatic_potential(
            basis, np.identity(88), grid_3d, coords, np.array([1, 2]), coord_type="spherical"
        ),
        horton_nucattract,
    )


def test_electrostatic_potential_mix():
    """Test gbasis.evals.esp.electorstatic_potential for mix coordinate type.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        evaluate_electrostatic_potential(
            basis, np.identity(88), grid_3d, coords, np.array([1, 2]), coord_type=["spherical"] * 9
        ),
        horton_nucattract,
    )
    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        evaluate_electrostatic_potential(
            basis, np.identity(103), grid_3d, coords, np.array([1, 2]), coord_type=["cartesian"] * 9
        ),
        horton_nucattract,
    )
