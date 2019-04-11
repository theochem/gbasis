"""Density Evaluation."""
from gbasis.eval import evaluate_basis_spherical_lincomb
import numpy as np


def eval_density_using_evaluated_orbs(one_density_matrix, orb_eval):
    """Return the evaluation of the density given the evaluated orbitals.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix.
    orb_eval : np.ndarray(K_orb, N)
        Orbitals evaluated at different grid points.
        The set of orbitals must be the same basis set used to build the one-electron density
        matrix.

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at different grid points.

    Raises
    ------
    TypeError
        If orb_eval is not a 2-dimensional numpy array with dtype float.
        If density_matrix is not a 2-dimensional numpy array with dtype float.
    ValueError
        If one-electron density matrix is not square.
        If the number of columns (or rows) of the one-electron density matrix is not equal to the
        number of rows of the orbital evaluations.

    """
    # test that inputs have the correct shape and type
    if not (
        isinstance(one_density_matrix, np.ndarray)
        and one_density_matrix.ndim == 2
        and one_density_matrix.dtype == float
    ):
        raise TypeError(
            "One-electron density matrix must be a two-dimensional numpy array with dtype float."
        )
    if not (isinstance(orb_eval, np.ndarray) and orb_eval.ndim == 2 and orb_eval.dtype == float):
        raise TypeError(
            "Evaluation of orbitals must be a two-dimensional numpy array with dtype float."
        )
    if one_density_matrix.shape[0] != one_density_matrix.shape[1]:
        raise ValueError("One-electron density matrix must be a square matrix.")
    if not np.allclose(one_density_matrix, one_density_matrix.T):
        raise ValueError("One-electron density matrix must be symmetric.")
    if one_density_matrix.shape[0] != orb_eval.shape[0]:
        raise ValueError(
            "Number of rows (and columns) of the density matrix must be equal to the number of rows"
            " of the orbital evaluations."
        )

    density = one_density_matrix.dot(orb_eval)
    density *= orb_eval
    return np.sum(density, axis=0)


def eval_density_using_basis(one_density_matrix, basis, coords, transform):
    """Return the density of the given transformed basis set at the given coordinates.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix.
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at different grid points.

    Notes
    -----
    The transformation matrix corresponds to the spherical contractions. If your transformation
    matrix corresponds to Cartesian contractions, please raise an issue.

    """
    orb_eval = evaluate_basis_spherical_lincomb(basis, coords, transform)
    return eval_density_using_evaluated_orbs(one_density_matrix, orb_eval)
