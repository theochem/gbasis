"""Test gbasis.integrals.angular_momentum."""

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import (
    _compute_differential_operator_integrals_intermediate,
)
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals_intermediate
from gbasis.integrals.angular_momentum import angular_momentum_integral, AngularMomentumIntegral
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_angular_momentum_construct_array_contraction(screen_basis, tol_screen):
    """Test integrals.angular_momentum.angular_momentumIntegral.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01]), "spherical"
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02]), "spherical"
    )

    # copied the code it is testing
    diff = _compute_differential_operator_integrals_intermediate(
        1, test_one.coord, 1, test_one.exps, test_two.coord, 2, test_two.exps
    )
    moment = _compute_multipole_moment_integrals_intermediate(
        np.zeros(3), 1, test_one.coord, 1, test_one.exps, test_two.coord, 2, test_two.exps
    )
    overlap = moment[0:1]

    norm_a = test_one.norm_prim_cart[:, None, :]
    norm_b = test_two.norm_prim_cart[:, :, None]
    int_px_dxx = (
        np.array(
            [
                overlap[0, 2, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 2, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 2, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 2, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 2, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 2, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[0]
    )
    int_px_dxy = (
        np.array(
            [
                overlap[0, 1, 1, 0, :, :] * moment[1, 1, 0, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 1, 1, 0, :, :] * diff[1, 1, 0, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 1, 1, 0, :, :] * overlap[0, 1, 0, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 1, 1, 0, :, :] * overlap[0, 1, 0, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 1, 1, 0, :, :] * diff[1, 1, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 1, 1, 0, :, :] * moment[1, 1, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[1]
    )
    int_px_dxz = (
        np.array(
            [
                overlap[0, 1, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 1, 0, 2, :, :]
                - overlap[0, 1, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 1, 0, 2, :, :],
                diff[1, 1, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 1, 0, 2, :, :]
                - moment[1, 1, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 1, 0, 2, :, :],
                moment[1, 1, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 1, 0, 2, :, :]
                - diff[1, 1, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 1, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[2]
    )
    int_px_dyy = (
        np.array(
            [
                overlap[0, 0, 1, 0, :, :] * moment[1, 2, 0, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 0, 1, 0, :, :] * diff[1, 2, 0, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 0, 1, 0, :, :] * overlap[0, 2, 0, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 0, 1, 0, :, :] * overlap[0, 2, 0, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 0, 1, 0, :, :] * diff[1, 2, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 0, 1, 0, :, :] * moment[1, 2, 0, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[3]
    )
    int_px_dyz = (
        np.array(
            [
                overlap[0, 0, 1, 0, :, :] * moment[1, 1, 0, 1, :, :] * diff[1, 1, 0, 2, :, :]
                - overlap[0, 0, 1, 0, :, :] * diff[1, 1, 0, 1, :, :] * moment[1, 1, 0, 2, :, :],
                diff[1, 0, 1, 0, :, :] * overlap[0, 1, 0, 1, :, :] * moment[1, 1, 0, 2, :, :]
                - moment[1, 0, 1, 0, :, :] * overlap[0, 1, 0, 1, :, :] * diff[1, 1, 0, 2, :, :],
                moment[1, 0, 1, 0, :, :] * diff[1, 1, 0, 1, :, :] * overlap[0, 1, 0, 2, :, :]
                - diff[1, 0, 1, 0, :, :] * moment[1, 1, 0, 1, :, :] * overlap[0, 1, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[4]
    )
    int_px_dzz = (
        np.array(
            [
                overlap[0, 0, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 2, 0, 2, :, :]
                - overlap[0, 0, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 2, 0, 2, :, :],
                diff[1, 0, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 2, 0, 2, :, :]
                - moment[1, 0, 1, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 2, 0, 2, :, :],
                moment[1, 0, 1, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 2, 0, 2, :, :]
                - diff[1, 0, 1, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 2, 0, 2, :, :],
            ]
        )
        * norm_a[0]
        * norm_b[5]
    )

    int_py_dxx = (
        np.array(
            [
                overlap[0, 2, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 2, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 2, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 2, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 2, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 2, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[0]
    )
    int_py_dxy = (
        np.array(
            [
                overlap[0, 1, 0, 0, :, :] * moment[1, 1, 1, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 1, 0, 0, :, :] * diff[1, 1, 1, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 1, 0, 0, :, :] * overlap[0, 1, 1, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 1, 0, 0, :, :] * overlap[0, 1, 1, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 1, 0, 0, :, :] * diff[1, 1, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 1, 0, 0, :, :] * moment[1, 1, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[1]
    )
    int_py_dxz = (
        np.array(
            [
                overlap[0, 1, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * diff[1, 1, 0, 2, :, :]
                - overlap[0, 1, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * moment[1, 1, 0, 2, :, :],
                diff[1, 1, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * moment[1, 1, 0, 2, :, :]
                - moment[1, 1, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * diff[1, 1, 0, 2, :, :],
                moment[1, 1, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * overlap[0, 1, 0, 2, :, :]
                - diff[1, 1, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * overlap[0, 1, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[2]
    )
    int_py_dyy = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 2, 1, 1, :, :] * diff[1, 0, 0, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 2, 1, 1, :, :] * moment[1, 0, 0, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 2, 1, 1, :, :] * moment[1, 0, 0, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 2, 1, 1, :, :] * diff[1, 0, 0, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 2, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 2, 1, 1, :, :] * overlap[0, 0, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[3]
    )
    int_py_dyz = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 1, 1, 1, :, :] * diff[1, 1, 0, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 1, 1, 1, :, :] * moment[1, 1, 0, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 1, 1, 1, :, :] * moment[1, 1, 0, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 1, 1, 1, :, :] * diff[1, 1, 0, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 1, 1, 1, :, :] * overlap[0, 1, 0, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 1, 1, 1, :, :] * overlap[0, 1, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[4]
    )
    int_py_dzz = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * diff[1, 2, 0, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * moment[1, 2, 0, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * moment[1, 2, 0, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 0, 1, 1, :, :] * diff[1, 2, 0, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 0, 1, 1, :, :] * overlap[0, 2, 0, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 0, 1, 1, :, :] * overlap[0, 2, 0, 2, :, :],
            ]
        )
        * norm_a[1]
        * norm_b[5]
    )

    int_pz_dxx = (
        np.array(
            [
                overlap[0, 2, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 0, 1, 2, :, :]
                - overlap[0, 2, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 0, 1, 2, :, :],
                diff[1, 2, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 0, 1, 2, :, :]
                - moment[1, 2, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 0, 1, 2, :, :],
                moment[1, 2, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :]
                - diff[1, 2, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[0]
    )
    int_pz_dxy = (
        np.array(
            [
                overlap[0, 1, 0, 0, :, :] * moment[1, 1, 0, 1, :, :] * diff[1, 0, 1, 2, :, :]
                - overlap[0, 1, 0, 0, :, :] * diff[1, 1, 0, 1, :, :] * moment[1, 0, 1, 2, :, :],
                diff[1, 1, 0, 0, :, :] * overlap[0, 1, 0, 1, :, :] * moment[1, 0, 1, 2, :, :]
                - moment[1, 1, 0, 0, :, :] * overlap[0, 1, 0, 1, :, :] * diff[1, 0, 1, 2, :, :],
                moment[1, 1, 0, 0, :, :] * diff[1, 1, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :]
                - diff[1, 1, 0, 0, :, :] * moment[1, 1, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[1]
    )
    int_pz_dxz = (
        np.array(
            [
                overlap[0, 1, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 1, 1, 2, :, :]
                - overlap[0, 1, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 1, 1, 2, :, :],
                diff[1, 1, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 1, 1, 2, :, :]
                - moment[1, 1, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 1, 1, 2, :, :],
                moment[1, 1, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 1, 1, 2, :, :]
                - diff[1, 1, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 1, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[2]
    )
    int_pz_dyy = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 2, 0, 1, :, :] * diff[1, 0, 1, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 2, 0, 1, :, :] * moment[1, 0, 1, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 2, 0, 1, :, :] * moment[1, 0, 1, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 2, 0, 1, :, :] * diff[1, 0, 1, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 2, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 2, 0, 1, :, :] * overlap[0, 0, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[3]
    )
    int_pz_dyz = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 1, 0, 1, :, :] * diff[1, 1, 1, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 1, 0, 1, :, :] * moment[1, 1, 1, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 1, 0, 1, :, :] * moment[1, 1, 1, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 1, 0, 1, :, :] * diff[1, 1, 1, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 1, 0, 1, :, :] * overlap[0, 1, 1, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 1, 0, 1, :, :] * overlap[0, 1, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[4]
    )
    int_pz_dzz = (
        np.array(
            [
                overlap[0, 0, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * diff[1, 2, 1, 2, :, :]
                - overlap[0, 0, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * moment[1, 2, 1, 2, :, :],
                diff[1, 0, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * moment[1, 2, 1, 2, :, :]
                - moment[1, 0, 0, 0, :, :] * overlap[0, 0, 0, 1, :, :] * diff[1, 2, 1, 2, :, :],
                moment[1, 0, 0, 0, :, :] * diff[1, 0, 0, 1, :, :] * overlap[0, 2, 1, 2, :, :]
                - diff[1, 0, 0, 0, :, :] * moment[1, 0, 0, 1, :, :] * overlap[0, 2, 1, 2, :, :],
            ]
        )
        * norm_a[2]
        * norm_b[5]
    )

    int_px_dxx = np.tensordot(int_px_dxx, test_one.coeffs, (2, 0))
    int_px_dxx = np.tensordot(int_px_dxx, test_two.coeffs, (1, 0))
    int_px_dxy = np.tensordot(int_px_dxy, test_one.coeffs, (2, 0))
    int_px_dxy = np.tensordot(int_px_dxy, test_two.coeffs, (1, 0))
    int_px_dxz = np.tensordot(int_px_dxz, test_one.coeffs, (2, 0))
    int_px_dxz = np.tensordot(int_px_dxz, test_two.coeffs, (1, 0))
    int_px_dyy = np.tensordot(int_px_dyy, test_one.coeffs, (2, 0))
    int_px_dyy = np.tensordot(int_px_dyy, test_two.coeffs, (1, 0))
    int_px_dyz = np.tensordot(int_px_dyz, test_one.coeffs, (2, 0))
    int_px_dyz = np.tensordot(int_px_dyz, test_two.coeffs, (1, 0))
    int_px_dzz = np.tensordot(int_px_dzz, test_one.coeffs, (2, 0))
    int_px_dzz = np.tensordot(int_px_dzz, test_two.coeffs, (1, 0))

    int_py_dxx = np.tensordot(int_py_dxx, test_one.coeffs, (2, 0))
    int_py_dxx = np.tensordot(int_py_dxx, test_two.coeffs, (1, 0))
    int_py_dxy = np.tensordot(int_py_dxy, test_one.coeffs, (2, 0))
    int_py_dxy = np.tensordot(int_py_dxy, test_two.coeffs, (1, 0))
    int_py_dxz = np.tensordot(int_py_dxz, test_one.coeffs, (2, 0))
    int_py_dxz = np.tensordot(int_py_dxz, test_two.coeffs, (1, 0))
    int_py_dyy = np.tensordot(int_py_dyy, test_one.coeffs, (2, 0))
    int_py_dyy = np.tensordot(int_py_dyy, test_two.coeffs, (1, 0))
    int_py_dyz = np.tensordot(int_py_dyz, test_one.coeffs, (2, 0))
    int_py_dyz = np.tensordot(int_py_dyz, test_two.coeffs, (1, 0))
    int_py_dzz = np.tensordot(int_py_dzz, test_one.coeffs, (2, 0))
    int_py_dzz = np.tensordot(int_py_dzz, test_two.coeffs, (1, 0))

    int_pz_dxx = np.tensordot(int_pz_dxx, test_one.coeffs, (2, 0))
    int_pz_dxx = np.tensordot(int_pz_dxx, test_two.coeffs, (1, 0))
    int_pz_dxy = np.tensordot(int_pz_dxy, test_one.coeffs, (2, 0))
    int_pz_dxy = np.tensordot(int_pz_dxy, test_two.coeffs, (1, 0))
    int_pz_dxz = np.tensordot(int_pz_dxz, test_one.coeffs, (2, 0))
    int_pz_dxz = np.tensordot(int_pz_dxz, test_two.coeffs, (1, 0))
    int_pz_dyy = np.tensordot(int_pz_dyy, test_one.coeffs, (2, 0))
    int_pz_dyy = np.tensordot(int_pz_dyy, test_two.coeffs, (1, 0))
    int_pz_dyz = np.tensordot(int_pz_dyz, test_one.coeffs, (2, 0))
    int_pz_dyz = np.tensordot(int_pz_dyz, test_two.coeffs, (1, 0))
    int_pz_dzz = np.tensordot(int_pz_dzz, test_one.coeffs, (2, 0))
    int_pz_dzz = np.tensordot(int_pz_dzz, test_two.coeffs, (1, 0))

    int_px_dxx = int_px_dxx.squeeze()
    int_px_dxy = int_px_dxy.squeeze()
    int_px_dxz = int_px_dxz.squeeze()
    int_px_dyy = int_px_dyy.squeeze()
    int_px_dyz = int_px_dyz.squeeze()
    int_px_dzz = int_px_dzz.squeeze()

    int_py_dxx = int_py_dxx.squeeze()
    int_py_dxy = int_py_dxy.squeeze()
    int_py_dxz = int_py_dxz.squeeze()
    int_py_dyy = int_py_dyy.squeeze()
    int_py_dyz = int_py_dyz.squeeze()
    int_py_dzz = int_py_dzz.squeeze()

    int_pz_dxx = int_pz_dxx.squeeze()
    int_pz_dxy = int_pz_dxy.squeeze()
    int_pz_dxz = int_pz_dxz.squeeze()
    int_pz_dyy = int_pz_dyy.squeeze()
    int_pz_dyz = int_pz_dyz.squeeze()
    int_pz_dzz = int_pz_dzz.squeeze()

    test = AngularMomentumIntegral.construct_array_contraction(
        test_one, test_two, screen_basis=screen_basis, tol_screen=tol_screen
    )
    assert test.shape == (1, 3, 1, 6, 3)
    assert np.allclose(
        test.squeeze(),
        -1j
        * np.array(
            [
                [int_px_dxx, int_px_dxy, int_px_dxz, int_px_dyy, int_px_dyz, int_px_dzz],
                [int_py_dxx, int_py_dxy, int_py_dxz, int_py_dyy, int_py_dyz, int_py_dzz],
                [int_pz_dxx, int_pz_dxy, int_pz_dxz, int_pz_dyy, int_pz_dyz, int_pz_dzz],
            ]
        ),
        atol=tol_screen,
    )

    with pytest.raises(TypeError):
        AngularMomentumIntegral.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        AngularMomentumIntegral.construct_array_contraction(None, test_two)


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_angular_momentum_integral_cartesian(screen_basis, tol_screen):
    """Test gbasis.integrals.angular_momentum.angular_momentum_integral_cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "cartesian")
    angular_momentum_integral_obj = AngularMomentumIntegral(basis)
    assert np.allclose(
        angular_momentum_integral_obj.construct_array_cartesian(
            screen_basis=screen_basis, tol_screen=tol_screen
        ),
        angular_momentum_integral(basis, screen_basis=False),
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_angular_momentum_integral_spherical(screen_basis, tol_screen):
    """Test gbasis.integrals.angular_momentum.angular_momentum_integral_spherical."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    angular_momentum_integral_obj = AngularMomentumIntegral(basis)
    assert np.allclose(
        angular_momentum_integral_obj.construct_array_spherical(
            screen_basis=screen_basis, tol_screen=tol_screen
        ),
        angular_momentum_integral(basis, screen_basis=False),
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_angular_momentum_integral_mix(screen_basis, tol_screen):
    """Test gbasis.integrals.angular_momentum.angular_momentum_integral_mix."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), ["spherical"] * 8)
    angular_momentum_integral_obj = AngularMomentumIntegral(basis)
    assert np.allclose(
        angular_momentum_integral_obj.construct_array_mix(
            ["spherical"] * 8, screen_basis=screen_basis, tol_screen=tol_screen
        ),
        angular_momentum_integral(basis, screen_basis=False),
        atol=tol_screen,
    )


@pytest.mark.parametrize("screen_basis, tol_screen", [(True, 1e-8), (False, 1e-12)])
def test_angular_momentum_integral_lincomb(screen_basis, tol_screen):
    """Test gbasis.integrals.angular_momentum.angular_momentum_integral_lincomb."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]), "spherical")
    angular_momentum_integral_obj = AngularMomentumIntegral(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        angular_momentum_integral_obj.construct_array_lincomb(
            transform, ["spherical"], screen_basis=screen_basis, tol_screen=tol_screen
        ),
        angular_momentum_integral(basis, transform, screen_basis=False),
        atol=tol_screen,
    )


@pytest.mark.parametrize("precision", [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
def test_angular_momentum_screening_accuracy(precision):
    """Test angular momentum screening."""

    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    atsymbols = ["H", "C", "Kr"]
    atcoords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    contraction = make_contractions(basis_dict, atsymbols, atcoords, "cartesian")

    #  the screening tolerance needs to be 1e-2 times the desired precision
    tol_screen = precision * 1e-2
    angular_momentum = angular_momentum_integral(contraction, tol_screen=tol_screen)
    angular_momentum_no_screen = angular_momentum_integral(contraction, screen_basis=False)
    assert np.allclose(angular_momentum, angular_momentum_no_screen, atol=precision)
