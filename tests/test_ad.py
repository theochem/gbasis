"""Test the ad module."""
import pytest
import numpy as np
from gbasis.gradient.ad import eval_nuc_deriv, eval_contractions


def test_eval_nuc_deriv():
    coords = np.random.rand(10, 3)
    center = np.random.rand(3)
    R_x, R_y, R_z = center
    angmom_comps = np.random.randint(0, 2, (1, 3))
    alphas = np.random.rand(1)

    prim_coeffs = np.random.rand(2, 1)
    norm = np.ones((1, 2))

    output = np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm)

    gradient = eval_nuc_deriv(coords, center, angmom_comps, alphas, prim_coeffs, norm)

    dx = 1e-5
    grad_numerical_x = (np.apply_along_axis(eval_contractions, 1, coords, R_x+dx, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x-dx, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)
    grad_numerical_y = (np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y+dx, R_z, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y-dx, R_z, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)
    grad_numerical_z = (np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y, R_z+dx, angmom_comps, alphas, prim_coeffs, norm) -\
                        np.apply_along_axis(eval_contractions, 1, coords, R_x, R_y, R_z-dx, angmom_comps, alphas, prim_coeffs, norm))/(2*dx)

    grad_numerical = np.hstack([grad_numerical_x, grad_numerical_y, grad_numerical_z])

    np.testing.assert_allclose(gradient, grad_numerical)