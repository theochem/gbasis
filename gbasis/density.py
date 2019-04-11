"""Density Evaluation."""
from gbasis.eval import evaluate_basis_spherical_lincomb
from gbasis.eval_deriv import evaluate_deriv_basis_spherical_lincomb
import numpy as np
from scipy.special import comb


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


def eval_deriv_density_using_basis(orders, one_density_matrix, basis, coords, transform):
    """Return the derivative of density of the given transformed basis set at the given coordinates.

    Parameters
    ----------
    orders : np.ndarray(3,)
        Orders of the derivative.
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
    density_deriv : np.ndarray(N,)
        Derivative of the density evaluated at different grid points.

    """
    # pylint: disable=R0914
    total_l_x, total_l_y, total_l_z = orders

    output = np.zeros(coords.shape[0])
    for l_x in range(total_l_x // 2 + 1):
        # prevent double counting for the middle of the even total_l_x
        # e.g. If total_l_x == 4, then l_x is in [0, 1, 2, 3, 4]. Exploiting symmetry we only need
        # to loop over [0, 1, 2] because l_x in [0, 4] and l_x in [1, 3] give the same result.
        # However, l_x = 2 needs to avoid double counting.
        if total_l_x % 2 == 0 and l_x == total_l_x / 2:
            factor = 1
        else:
            factor = 2
        for l_y in range(total_l_y + 1):
            for l_z in range(total_l_z + 1):
                num_occurence = comb(total_l_x, l_x) * comb(total_l_y, l_y) * comb(total_l_z, l_z)
                orders_one = np.array([l_x, l_y, l_z])
                orders_two = orders - orders_one
                deriv_orb_eval_one = evaluate_deriv_basis_spherical_lincomb(
                    basis, coords, orders_one, transform
                )
                deriv_orb_eval_two = evaluate_deriv_basis_spherical_lincomb(
                    basis, coords, orders_two, transform
                )
                density = one_density_matrix.dot(deriv_orb_eval_two)
                density *= deriv_orb_eval_one
                density = np.sum(density, axis=0)
                output += factor * num_occurence * density
    return output
