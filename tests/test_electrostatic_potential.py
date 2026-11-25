"""Tests for gbasis.evals.electrostatic_potential."""
from gbasis.evals.electrostatic_potential import electrostatic_potential
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_electrostatic_potential():
    """Test gbasis.evals.electrostatic_potential.electorstatic_potential.

    Tested by using point_charge.point_charge_cartesian.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    cartesian_basis = make_contractions(basis_dict, ["H", "He"], coords, "cartesian")
    cartesian_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in cartesian_basis
    ]
    spherical_basis = make_contractions(basis_dict, ["H", "He"], coords, "spherical")
    spherical_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in spherical_basis
    ]
    mixed_basis = make_contractions(basis_dict, ["H", "He"], coords, ["spherical"] * 9)
    mixed_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in mixed_basis
    ]

    # check density_matrix type
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103).tolist(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103).flatten(),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )
    # check nuclear_coords
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3).tolist(),
            np.array([1, 2]),
        )
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 4),
            np.array([1, 2]),
        )
    # check nuclear charges
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).reshape(1, 2),
        )
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]).tolist(),
        )
    # check density_matrix symmetry
    with pytest.raises(ValueError):
        electrostatic_potential(
            cartesian_basis,
            np.eye(103, 102),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )
    # check nuclear_coords and nuclear_charges shapes
    with pytest.raises(ValueError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(3, 3),
            np.array([1, 2]),
        )
    # check threshold_dist types
    with pytest.raises(TypeError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            threshold_dist=None,
        )
    # check threshold_dist value
    with pytest.raises(ValueError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
            threshold_dist=-0.1,
        )
    with pytest.raises(ValueError):
        electrostatic_potential(
            cartesian_basis,
            np.identity(88),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )
    with pytest.raises(ValueError):
        electrostatic_potential(
            spherical_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )
    with pytest.raises(ValueError):
        electrostatic_potential(
            mixed_basis,
            np.identity(103),
            np.random.rand(10, 3),
            np.random.rand(2, 3),
            np.array([1, 2]),
        )


def test_electrostatic_potential_cartesian():
    """Test gbasis.evals.electrostatic_potential.electorstatic_potential_cartesian.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords, "cartesian")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        electrostatic_potential(basis, np.identity(103), grid_3d, coords, np.array([1, 2])),
        horton_nucattract,
    )


def test_electrostatic_potential_spherical():
    """Test gbasis.evals.electrostatic_potential.electorstatic_potential_spherical.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        electrostatic_potential(basis, np.identity(88), grid_3d, coords, np.array([1, 2])),
        horton_nucattract,
    )


def test_electrostatic_potential_mix():
    """Test gbasis.evals.electrostatic_potential.electorstatic_potential_mix.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.
    Density matrix is an identity matrix.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    spherical_basis = make_contractions(basis_dict, ["H", "He"], coords, ["spherical"] * 9)
    spherical_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in spherical_basis
    ]
    cartesian_basis = make_contractions(basis_dict, ["H", "He"], coords, ["cartesian"] * 9)
    cartesian_basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in cartesian_basis
    ]

    grid_1d = np.linspace(-2, 2, num=5)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    horton_nucattract = np.load(find_datafile("data_horton_hhe_sph_esp.npy"))
    assert np.allclose(
        electrostatic_potential(
            spherical_basis, np.identity(88), grid_3d, coords, np.array([1, 2])
        ),
        horton_nucattract,
    )
    horton_nucattract = np.load(find_datafile("data_horton_hhe_cart_esp.npy"))
    assert np.allclose(
        electrostatic_potential(
            cartesian_basis, np.identity(103), grid_3d, coords, np.array([1, 2])
        ),
        horton_nucattract,
    )


def test_electrostatic_potential_threshold_dist():
    """Test threshold_dist properly discards nuclei based on Euclidean distance."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    # Diatomic with C (Z=6) at origin and H (Z=1) at (2.0, 0, 0)
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    charges = np.array([6.0, 1.0])
    basis = make_contractions(basis_dict, ["C", "H"], coords, "cartesian")
    basis = [
        HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type)
        for i in basis
    ]
    dm = np.identity(sum(c.num_cart for c in basis))

    # Point 1: at (1.5, 0, 0), distances are 1.5 to C and 0.5 to H
    # With threshold_dist = 0.4, neither nucleus should be discarded
    pt_mid = np.array([[1.5, 0.0, 0.0]])
    esp_full = electrostatic_potential(basis, dm, pt_mid, coords, charges, threshold_dist=0.0)
    esp_thresh_none = electrostatic_potential(
        basis, dm, pt_mid, coords, charges, threshold_dist=0.4
    )
    assert np.allclose(esp_full, esp_thresh_none)

    # Point 2: at (0.2, 0, 0), distance is 0.2 to C and 1.8 to H
    # With threshold_dist = 0.5, C (dist=0.2 < 0.5) is discarded, H (dist=1.8 > 0.5) is kept
    pt_near_c = np.array([[0.2, 0.0, 0.0]])
    esp_c_full = electrostatic_potential(
        basis, dm, pt_near_c, coords, charges, threshold_dist=0.0
    )
    esp_c_thresh = electrostatic_potential(
        basis, dm, pt_near_c, coords, charges, threshold_dist=0.5
    )
    # Difference should be exactly C's nuclear contribution: 6.0 / 0.2 = 30.0 Ha
    assert np.allclose(esp_c_full - esp_c_thresh, np.array([30.0]))

    # Point 3: at (1.9, 0, 0), distance is 1.9 to C and 0.1 to H
    # With threshold_dist = 0.5, H (dist=0.1 < 0.5) is discarded, C (dist=1.9 > 0.5) is kept
    pt_near_h = np.array([[1.9, 0.0, 0.0]])
    esp_h_full = electrostatic_potential(
        basis, dm, pt_near_h, coords, charges, threshold_dist=0.0
    )
    esp_h_thresh = electrostatic_potential(
        basis, dm, pt_near_h, coords, charges, threshold_dist=0.5
    )
    # Difference should be exactly H's nuclear contribution: 1.0 / 0.1 = 10.0 Ha
    assert np.allclose(esp_h_full - esp_h_thresh, np.array([10.0]))
