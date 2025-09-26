"""Test gbasis.evals.density."""

from gbasis.evals.density import (
    evaluate_density,
    evaluate_density_gradient,
    evaluate_density_hessian,
    evaluate_density_laplacian,
    evaluate_density_using_evaluated_orbs,
    evaluate_deriv_density,
    evaluate_general_kinetic_energy_density,
    evaluate_posdef_kinetic_energy_density,
)
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_evaluate_density_using_evaluated_orbs():
    """Test gbasis.evals.density.evaluate_density_using_evaluated_orbs."""
    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0], [2.0]])
    dens = evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    assert np.all(dens >= 0.0)
    assert np.allclose(dens, np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval))

    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dens = evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    assert np.all(dens >= 0)
    assert np.allclose(dens, np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval))

    with pytest.raises(TypeError):
        orb_eval = [[1.0, 2.0], [1.0, 2.0]]
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        orb_eval = np.array([[1, 2], [1, 2]], dtype=bool)
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)

    orb_eval = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(TypeError):
        density_mat = [[1.0, 2.0], [1.0, 2.0]]
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([[1, 2], [1, 2]], dtype=bool)
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([1.0, 2.0, 3.0])
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        evaluate_density_using_evaluated_orbs(density_mat, orb_eval)


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
def test_evaluate_density(screen_basis, tol_screen):
    """Test gbasis.evals.density.evaluate_density."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    points = np.random.rand(10, 3)

    evaluate_orbs = evaluate_basis(
        basis, points, transform, screen_basis=screen_basis, tol_screen=tol_screen
    )
    dens = evaluate_density(
        density, basis, points, transform, screen_basis=screen_basis, tol_screen=tol_screen
    )
    assert np.all(dens >= 0.0)
    assert np.allclose(
        dens, np.einsum("ij,ik,jk->k", density, evaluate_orbs, evaluate_orbs), atol=tol_screen
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_deriv_density(screen_basis, tol_screen, deriv_type):
    """Test gbasis.evals.density.evaluate_deriv_density."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    points = np.random.rand(10, 3)

    # check non screened x derivative against screened y derivative
    assert np.allclose(
        evaluate_deriv_density(
            np.array([1, 0, 0]),
            density,
            basis,
            points,
            transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform, screen_basis=False),
            evaluate_basis(basis, points, transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis(basis, points, transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform, screen_basis=False),
        ),
        atol=tol_screen,
    )

    # check non screened y derivative against screened y derivative
    assert np.allclose(
        evaluate_deriv_density(
            np.array([0, 1, 0]),
            density,
            basis,
            points,
            transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform, screen_basis=False),
            evaluate_basis(basis, points, transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis(basis, points, transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform, screen_basis=False),
        ),
        atol=tol_screen,
    )

    assert np.allclose(
        evaluate_deriv_density(
            np.array([0, 0, 1]),
            density,
            basis,
            points,
            transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 0, 1]), transform, screen_basis=False),
            evaluate_basis(basis, points, transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis(basis, points, transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 0, 1]), transform, screen_basis=False),
        ),
        atol=tol_screen,
    )

    assert np.allclose(
        evaluate_deriv_density(
            np.array([2, 3, 0]),
            density,
            basis,
            points,
            transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 0, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([2, 3, 0]), transform, screen_basis=False),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([2, 2, 0]), transform, screen_basis=False),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 2, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([2, 1, 0]), transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([0, 3, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([2, 0, 0]), transform, screen_basis=False),
        )
        + 2
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([1, 3, 0]), transform, screen_basis=False),
        )
        + 2
        * 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([1, 1, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([1, 2, 0]), transform, screen_basis=False),
        )
        + 2
        * 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([1, 2, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([1, 1, 0]), transform, screen_basis=False),
        )
        + 2
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([1, 3, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([2, 0, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 3, 0]), transform, screen_basis=False),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([2, 1, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 2, 0]), transform, screen_basis=False),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([2, 2, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform, screen_basis=False),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis(basis, points, np.array([2, 3, 0]), transform, screen_basis=False),
            evaluate_deriv_basis(basis, points, np.array([0, 0, 0]), transform, screen_basis=False),
        ),
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_density_gradient(screen_basis, tol_screen, deriv_type):
    """Test gbasis.evals.density.evaluate_density_gradient."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    points = np.random.rand(10, 3)

    np.allclose(
        evaluate_density_gradient(
            density,
            basis,
            points,
            transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ).T,
        np.array(
            [
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform),
                    evaluate_basis(basis, points, transform, screen_basis=False),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis(basis, points, transform, screen_basis=False),
                    evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform),
                ),
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform),
                    evaluate_basis(basis, points, transform, screen_basis=False),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis(basis, points, transform, screen_basis=False),
                    evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform),
                ),
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis(basis, points, np.array([0, 0, 1]), transform),
                    evaluate_basis(basis, points, transform, screen_basis=False),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis(basis, points, transform, screen_basis=False),
                    evaluate_deriv_basis(basis, points, np.array([0, 0, 1]), transform),
                ),
            ]
        ),
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
def test_evaluate_density_horton(screen_basis, tol_screen):
    """Test gbasis.evals.density.evaluate_density against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density = np.load(find_datafile("data_horton_hhe_sph_density.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    dens = evaluate_density(
        np.identity(88),
        basis,
        grid_3d,
        np.identity(88),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    assert np.all(dens >= 0.0)
    assert np.allclose(dens, horton_density, atol=tol_screen)


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_density_gradient_horton(screen_basis, tol_screen, deriv_type):
    """Test gbasis.evals.density.evaluate_density_gradient against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density_gradient = np.load(find_datafile("data_horton_hhe_sph_density_gradient.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        evaluate_density_gradient(
            np.identity(88),
            basis,
            grid_3d,
            np.identity(88),
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        horton_density_gradient,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_hessian_deriv_horton(screen_basis, tol_screen, deriv_type):
    """Test gbasis.evals.density.evaluate_density_hessian against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density_hessian = np.zeros((10**3, 3, 3))
    horton_density_hessian[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] = np.load(
        find_datafile("data_horton_hhe_sph_density_hessian.npy")
    )
    horton_density_hessian[:, [1, 2, 2], [0, 0, 1]] = horton_density_hessian[
        :, [0, 0, 1], [1, 2, 2]
    ]

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        evaluate_density_hessian(
            np.identity(88),
            basis,
            grid_3d,
            np.identity(88),
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        horton_density_hessian,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_laplacian_deriv_horton(screen_basis, tol_screen, deriv_type):
    """Test gbasis.evals.density.evaluate_density_laplacian against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density_laplacian = np.load(find_datafile("data_horton_hhe_sph_density_laplacian.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        evaluate_density_laplacian(
            np.identity(88),
            basis,
            grid_3d,
            np.identity(88),
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        horton_density_laplacian,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_posdef_kinetic_energy_density(screen_basis, tol_screen, deriv_type):
    """Test evaluate_posdef_kinetic_energy_density against results from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density_kinetic_density = np.load(
        find_datafile("data_horton_hhe_sph_posdef_kinetic_density.npy")
    )

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    dens = evaluate_posdef_kinetic_energy_density(
        np.identity(88),
        basis,
        grid_3d,
        np.identity(88),
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    assert np.all(dens >= 0.0)
    assert np.allclose(dens, horton_density_kinetic_density, atol=tol_screen)


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_general_kinetic_energy_density_horton(screen_basis, tol_screen, deriv_type):
    """Test evaluate_general_kinetic_energy_density against results from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    It's actually just testing posdef part.

    """
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    points = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], points, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    horton_density_kinetic_density = np.load(
        find_datafile("data_horton_hhe_sph_posdef_kinetic_density.npy")
    )

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        evaluate_general_kinetic_energy_density(
            np.identity(88),
            basis,
            grid_3d,
            0,
            np.identity(88),
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        horton_density_kinetic_density,
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
@pytest.mark.parametrize("deriv_type", ["general", "direct"])
def test_evaluate_general_kinetic_energy_density(screen_basis, tol_screen, deriv_type):
    """Test density.evaluate_general_kinetic_energy_density."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    points = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], points, "spherical")
    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        evaluate_general_kinetic_energy_density(
            np.identity(40), basis, points, np.identity(40), np.array(0)
        )
    with pytest.raises(TypeError):
        evaluate_general_kinetic_energy_density(
            np.identity(40),
            basis,
            points,
            None,
            np.identity(40),
        )
    assert np.allclose(
        evaluate_general_kinetic_energy_density(
            np.identity(40),
            basis,
            points,
            1,
            np.identity(40),
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ),
        evaluate_posdef_kinetic_energy_density(
            np.identity(40), basis, points, np.identity(40), screen_basis=False
        )
        + evaluate_density_laplacian(
            np.identity(40), basis, points, np.identity(40), screen_basis=False
        ),
        atol=tol_screen,
    )
