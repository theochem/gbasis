"""Test gbasis.density."""
from gbasis.density import (
    eval_density,
    eval_density_gradient,
    eval_density_hessian,
    eval_density_laplacian,
    eval_density_using_evaluated_orbs,
    eval_deriv_density,
    eval_posdef_kinetic_energy_density,
)
from gbasis.eval import evaluate_basis_lincomb
from gbasis.eval_deriv import evaluate_deriv_basis_lincomb
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_eval_density_using_evaluated_orbs():
    """Test gbasis.density.eval_density_using_evaluated_orbs."""
    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0], [2.0]])
    assert np.allclose(
        eval_density_using_evaluated_orbs(density_mat, orb_eval),
        np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval),
    )
    density_mat = np.array([[1.0, 2.0], [2.0, 3.0]])
    orb_eval = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(
        eval_density_using_evaluated_orbs(density_mat, orb_eval),
        np.einsum("ij,ik,jk->k", density_mat, orb_eval, orb_eval),
    )

    with pytest.raises(TypeError):
        orb_eval = [[1.0, 2.0], [1.0, 2.0]]
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        orb_eval = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density_using_evaluated_orbs(density_mat, orb_eval)

    orb_eval = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(TypeError):
        density_mat = [[1.0, 2.0], [1.0, 2.0]]
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([[1, 2], [1, 2]], dtype=bool)
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(TypeError):
        density_mat = np.array([1.0, 2.0, 3.0])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)
    with pytest.raises(ValueError):
        density_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        eval_density_using_evaluated_orbs(density_mat, orb_eval)


def test_eval_density():
    """Test gbasis.density.eval_density."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    coords = np.random.rand(10, 3)

    eval_orbs = evaluate_basis_lincomb(basis, coords, transform)
    assert np.allclose(
        eval_density(density, basis, coords, transform),
        np.einsum("ij,ik,jk->k", density, eval_orbs, eval_orbs),
    )


def test_eval_deriv_density():
    """Test gbasis.density.eval_deriv_density."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    coords = np.random.rand(10, 3)

    assert np.allclose(
        eval_deriv_density(np.array([1, 0, 0]), density, basis, coords, transform),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
            evaluate_basis_lincomb(basis, coords, transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis_lincomb(basis, coords, transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
        ),
    )

    assert np.allclose(
        eval_deriv_density(np.array([0, 1, 0]), density, basis, coords, transform),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
            evaluate_basis_lincomb(basis, coords, transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis_lincomb(basis, coords, transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
        ),
    )

    assert np.allclose(
        eval_deriv_density(np.array([0, 0, 1]), density, basis, coords, transform),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 1]), transform),
            evaluate_basis_lincomb(basis, coords, transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_basis_lincomb(basis, coords, transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 1]), transform),
        ),
    )

    assert np.allclose(
        eval_deriv_density(np.array([2, 3, 0]), density, basis, coords, transform),
        np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 3, 0]), transform),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 2, 0]), transform),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 2, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 1, 0]), transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 3, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 0, 0]), transform),
        )
        + 2
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 3, 0]), transform),
        )
        + 2
        * 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 1, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 2, 0]), transform),
        )
        + 2
        * 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 2, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 1, 0]), transform),
        )
        + 2
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 3, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 0, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 3, 0]), transform),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 1, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 2, 0]), transform),
        )
        + 3
        * np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 2, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
        )
        + np.einsum(
            "ij,ik,jk->k",
            density,
            evaluate_deriv_basis_lincomb(basis, coords, np.array([2, 3, 0]), transform),
            evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 0]), transform),
        ),
    )


def test_eval_density_gradient():
    """Test gbasis.density.eval_density_gradient."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    transform = np.random.rand(14, 18)
    density = np.random.rand(14, 14)
    density += density.T
    coords = np.random.rand(10, 3)

    np.allclose(
        eval_density_gradient(density, basis, coords, transform).T,
        np.array(
            [
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
                    evaluate_basis_lincomb(basis, coords, transform),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis_lincomb(basis, coords, transform),
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([1, 0, 0]), transform),
                ),
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
                    evaluate_basis_lincomb(basis, coords, transform),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis_lincomb(basis, coords, transform),
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 1, 0]), transform),
                ),
                np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 1]), transform),
                    evaluate_basis_lincomb(basis, coords, transform),
                )
                + np.einsum(
                    "ij,ik,jk->k",
                    density,
                    evaluate_basis_lincomb(basis, coords, transform),
                    evaluate_deriv_basis_lincomb(basis, coords, np.array([0, 0, 1]), transform),
                ),
            ]
        ),
    )


def test_eval_density_horton():
    """Test gbasis.density.eval_density against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_density = np.load(find_datafile("data_horton_hhe_sph_density.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        eval_density(np.identity(88), basis, grid_3d, np.identity(88)), horton_density
    )


def test_eval_density_gradient_horton():
    """Test gbasis.density.eval_density_gradient against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_density_gradient = np.load(find_datafile("data_horton_hhe_sph_density_gradient.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        eval_density_gradient(np.identity(88), basis, grid_3d, np.identity(88)),
        horton_density_gradient,
    )


def test_eval_hessian_deriv_horton():
    """Test gbasis.density.eval_density_hessian against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_density_hessian = np.zeros((10 ** 3, 3, 3))
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
        eval_density_hessian(np.identity(88), basis, grid_3d, np.identity(88)),
        horton_density_hessian,
    )


def test_eval_laplacian_deriv_horton():
    """Test gbasis.density.eval_density_laplacian against result from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_density_laplacian = np.load(find_datafile("data_horton_hhe_sph_density_laplacian.npy"))

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        eval_density_laplacian(np.identity(88), basis, grid_3d, np.identity(88)),
        horton_density_laplacian,
    )


def test_eval_posdef_kinetic_energy_density():
    """Test gbasis.kinetic_energy.eval_posdef_kinetic_energy_density against results from HORTON.

    The test case is diatomic with H and He separated by 0.8 angstroms with basis set ANO-RCC.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    # NOTE: used HORTON's conversion factor for angstroms to bohr
    coords = np.array([[0, 0, 0], [0.8 * 1.0 / 0.5291772083, 0, 0]])
    basis = make_contractions(basis_dict, ["H", "He"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    horton_density_kinetic_density = np.load(
        find_datafile("data_horton_hhe_sph_posdef_kinetic_density.npy")
    )

    grid_1d = np.linspace(-2, 2, num=10)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d)
    grid_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    assert np.allclose(
        eval_posdef_kinetic_energy_density(np.identity(88), basis, grid_3d, np.identity(88)),
        horton_density_kinetic_density,
    )
