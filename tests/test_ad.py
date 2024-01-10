"""Test the ad module."""
import pytest
import numpy as np
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.gradient.ad import eval_nuc_deriv, _eval_nuc_deriv, eval_contractions


def test_eval_nuc_deriv():
    """Test the _eval_nuc_deriv function."""
    coords = np.random.rand(10, 3)
    center = np.random.rand(3)
    R_x, R_y, R_z = center
    angmom_comps = np.random.randint(0, 2, (1, 3))
    alphas = np.random.rand(1)

    prim_coeffs = np.random.rand(2, 1)
    norm = np.ones((1, 2))
    orders = np.array([1, 1, 1])

    gradient = _eval_nuc_deriv(coords, orders, center, angmom_comps, alphas, prim_coeffs, norm)

    dx = 1e-5
    grad_numerical_x = (np.apply_along_axis(eval_contractions, 1, coords, R_x+dx, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x-dx, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)
    grad_numerical_y = (np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y+dx, R_z, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y-dx, R_z, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)
    grad_numerical_z = (np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y, R_z+dx, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y, R_z-dx, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)

    grad_numerical = np.hstack([grad_numerical_x, grad_numerical_y, grad_numerical_z])

    np.testing.assert_allclose(gradient, grad_numerical)

def test_eval_nuc_deriv_full():
    """Test the eval_nuc_deriv function."""
    coords = np.random.rand(10, 3)
    center = np.random.rand(3)

    R_x, R_y, R_z = center
    angmom_comps = np.random.randint(0, 2, (2, 3))
    alphas = np.random.rand(4)

    prim_coeffs = np.random.rand(4, 5)
    norms = np.ones((2, 4))

    orders = np.array([1, 1, 1])
    output = eval_nuc_deriv(coords, orders, center, angmom_comps, alphas, prim_coeffs, norms)

    for i in range(3):
        dx = np.zeros(3)
        dx[i] = 1e-5
        grad_numerical = (_eval_deriv_contractions(coords, np.array([0, 0, 0]), center+dx, angmom_comps, alphas, prim_coeffs, norms) -\
            _eval_deriv_contractions(coords, np.array([0, 0, 0]), center-dx, angmom_comps, alphas, prim_coeffs, norms))/(2*dx[i])
        np.allclose(output[:, :, :, i], grad_numerical)

    # testing second order derivative
    orders = np.array([2, 2, 2])
    output = eval_nuc_deriv(coords, orders, center, angmom_comps, alphas, prim_coeffs, norms)
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = 1e-5
        grad_numerical = (_eval_deriv_contractions(coords, np.array([0, 0, 0]), center+dx, angmom_comps, alphas, prim_coeffs, norms) +\
            _eval_deriv_contractions(coords, np.array([0, 0, 0]), center-dx, angmom_comps, alphas, prim_coeffs, norms) -\
            2*_eval_deriv_contractions(coords, np.array([0, 0, 0]), center, angmom_comps, alphas, prim_coeffs, norms))/(dx[i]**2)
        np.allclose(output[:, :, :, i], grad_numerical)