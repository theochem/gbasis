"""Density Evaluation."""
import numpy as np


def eval_density(density_mat, orb_eval):
    """Return the evaluation of the density at different points.

    Parameters
    ----------
    density_mat : np.ndarray(L, L)
        One electron density matrix.
    orb_eval : {np.ndarray(L), np.ndarray(L, N), np.ndarray(L, N, N, N)}
        Array of orbitals evaluated at different grid points.
        The set of orbitals must be the same basis set used to build the charge density matrix.

    Returns
    -------
    total_density : {np.ndarray(N, 1), np.ndarray(N, N, N)}
        Density evaluated at different grid points.
        Dimension of the density will match the dimension the grid part of orb_eval.

    Raises
    ------
    TypeError
        If orb_eval is not an array of floats.
        If density_mat is not an array of floats.
    ValueError
        If density_mat is not a 2-dimensional matrix.
        If first dimension of density_mat and orb_eval are not equal.

    """
    # test that inputs have the correct shape and type
    if not (isinstance(orb_eval, np.ndarray) and orb_eval.dtype == float):
        raise TypeError("Evaluation of orbitals must be a numpy array of data type float.")
    if not (isinstance(density_mat, np.ndarray) and density_mat.dtype == float):
        raise TypeError("Density matrix must be a numpy array of data type float.")
    if density_mat.ndim != 2 or density_mat.shape[0] != density_mat.shape[1]:
        raise ValueError("Density matrix must be a square matrix.")
    if density_mat.shape[0] != orb_eval.shape[0]:
        raise ValueError("Size of 1st dimension of the density matrix and orbitals must match.")

    # resize orbital array to 2 dimensions
    old_shape = orb_eval.shape
    orb_eval.shape = (orb_eval.shape[0], np.prod(old_shape[1:]))

    # evaluate partial density contributions
    partial_density = (
        orb_eval[np.newaxis, :] * orb_eval[:, np.newaxis] * density_mat[:, :, np.newaxis]
    )

    # sum over all orbital pairs
    total_density = partial_density.sum(axis=(0, 1))

    # resize density array
    total_density.shape = old_shape[1:]

    return total_density
