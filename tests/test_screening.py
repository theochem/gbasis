"""Test gbasis.screening"""

import numpy as np
import pytest
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.screening import is_two_index_overlap_screened, compute_primitive_upper_bound
from gbasis.screening import compute_primitive_cutoff_radius, compute_contraction_upper_bound, compute_contraction_upper_bond
from gbasis.evals.eval_deriv import _eval_deriv_contractions
from gbasis.utils import factorial2
from utils import find_datafile


def get_atom_contractions_data(atsym, atcoords):
    """Get the STO-6G contractions for a given atom symbol and coordinates."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, [atsym], atcoords, "cartesian")
    return basis


@pytest.mark.parametrize("bond_length", [0, 0.999, 2.0, 4.0, 8.0, 50.0, 100.0])
@pytest.mark.parametrize("tol_screen", [1e-4, 1e-8, 1e-12])
def test_is_two_index_overlap_screened(bond_length, tol_screen):
    contractions_one = get_atom_contractions_data("H", np.array([[0, 0, 0]]))
    contractions_two = get_atom_contractions_data("O", np.array([[0, 0, bond_length]]))
    screen_pairs_list = []
    screening_cutoffs = []
    for h_shell in contractions_one:
        for o_shell in contractions_two:
            screen_pairs_list.append(is_two_index_overlap_screened(h_shell, o_shell, tol_screen))
            alpha_a = min(h_shell.exps)
            alpha_b = min(o_shell.exps)
            cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(tol_screen))
            screening_cutoffs.append(cutoff)

    # bonds too close, not any pairs should be screened
    if bond_length < 1.0:
        assert not any(screen_pairs_list)
    # bonds too far, all pairs should be screened
    elif bond_length > 30.0:
        assert all(screen_pairs_list)
    # intermediate bond lengths, some pairs should be screened
    else:
        ref_screened = np.array(screening_cutoffs) < bond_length
        assert np.all(
            np.array(screen_pairs_list) == ref_screened
        ), "Screening results do not match the expected values based on cutoff distances."


@pytest.mark.parametrize("angm", [0, 3])
@pytest.mark.parametrize("alpha", [0.5, 2.0])
@pytest.mark.parametrize("coeff", [0.1, 0.8])
@pytest.mark.parametrize("deriv_order", [0, 4])
@pytest.mark.parametrize("tol_screen", [1e-4, 1e-8])
def test_compute_primitive_cutoff_radius(angm, alpha, coeff, deriv_order, tol_screen):
    """Test the computation of the primitive cutoff radius."""

    def compute_primitive_value(r, c, alpha, angm, deriv_order):
        """Compute the primitive value at the given radius."""
        n = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (angm / 2)
        # Include derivative polynomial factor as radial bound
        radial_factor = r ** (angm + deriv_order)
        derivative_scale = (2 * alpha) ** deriv_order  # optional scaling from derivative
        return c * n * derivative_scale * radial_factor * np.exp(-alpha * r**2)

    cutoff_r = compute_primitive_cutoff_radius(
        coeff, alpha, angm, deriv_order=deriv_order, tol_screen=tol_screen
    )
    val_over_cutoff = compute_primitive_value(
        cutoff_r + 1e-3, coeff, alpha, angm, deriv_order=deriv_order
    )

    assert (
        val_over_cutoff < tol_screen
    ), f"Value {val_over_cutoff} at r={cutoff_r + 1e-3} is not below the tolerance {tol_screen}"


def test_compute_compute_primitive_upper_bound():
    """Test the computation of the primitive upper bound."""

    rgrid = np.linspace(0, 80, 1000)

    def compute_primitive_value(r, c, alpha, angm, deriv_order):
        """Compute the primitive value at the given radius."""
        n = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (angm / 2)
        # Include derivative polynomial factor as radial bound
        radial_factor = r ** (angm + deriv_order)
        derivative_scale = (2 * alpha) ** deriv_order  # optional scaling from derivative
        return c * n * derivative_scale * radial_factor * np.exp(-alpha * r**2)

    angmom = 3
    deriv_order = 2
    alpha = 0.5
    coeff = 0.8
    prim_vals = compute_primitive_value(rgrid, coeff, alpha, angmom, deriv_order)
    upper_bound = compute_primitive_upper_bound(coeff, alpha, angmom, deriv_order)
    assert np.all(prim_vals <= upper_bound), "Primitive values exceed the computed upper bound"


@pytest.mark.parametrize("deriv_order", [0, 1, 2])
def test_compute_contraction_upper_bound(deriv_order):
    """Test the computation of the contraction upper bound."""

    atcoord = np.array([[0.0, 0.0, 0.0]])
    max_length = 80.0
    npoints = 200
    # Create points along multiple axes and diagonals
    points_z = np.array([[0.0, 0.0, 1.0]]) * np.linspace(1e-8, max_length, npoints)[:, None]
    points_x = np.array([[1.0, 0.0, 0.0]]) * np.linspace(1e-8, max_length, npoints)[:, None]
    points_diag = (np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3)) * np.linspace(1e-8, max_length, npoints)[:, None]
    points = np.vstack([points_z, points_x, points_diag])

    # Test all derivative partitions of deriv_order across (x, y, z)
    partitions = []
    for kx in range(deriv_order + 1):
        for ky in range(deriv_order - kx + 1):
            kz = deriv_order - kx - ky
            partitions.append(np.array([kx, ky, kz]))

    basis = get_atom_contractions_data("O", atcoord)
    for contractions in basis:
        computed_upper_bound = compute_contraction_upper_bound(contractions, deriv_order=deriv_order)
        # Verify deprecated alias returns the same value
        alias_upper_bound = compute_contraction_upper_bond(contractions, deriv_order=deriv_order)
        assert np.isclose(computed_upper_bound, alias_upper_bound)

        alphas = contractions.exps
        prim_coeffs = contractions.coeffs
        angmom_comps = contractions.angmom_components_cart
        center = contractions.coord
        norm_prim_cart = contractions.norm_prim_cart

        for orders in partitions:
            contraction_values = _eval_deriv_contractions(
                points, orders, center, angmom_comps, alphas, prim_coeffs, norm_prim_cart
            )
            max_contraction_values = np.max(np.abs(contraction_values))
            assert max_contraction_values <= computed_upper_bound, (
                f"Contraction values exceed the computed upper bound: "
                f"{max_contraction_values} > {computed_upper_bound}"
            )


def test_compute_primitive_cutoff_radius_array_inputs():
    """Test compute_primitive_cutoff_radius with array inputs where some ratios >= 1."""
    c = np.array([1.0, 0.01])
    alpha = np.array([1.0, 1.0])

    # angm = 0
    r0 = compute_primitive_cutoff_radius(c, alpha, angm=0, deriv_order=0, tol_screen=0.5)
    assert not np.any(np.isnan(r0))
    assert r0[1] == 0.0
    assert r0[0] > 0.0

    # angm = 1
    r1 = compute_primitive_cutoff_radius(c, alpha, angm=1, deriv_order=0, tol_screen=0.5)
    assert not np.any(np.isnan(r1))
    assert r1[1] == 0.0
    assert r1[0] > 0.0

    # 2D array input (e.g. from contractions.coeffs of shape (K, M))
    c_2d = np.array([[1.0, 0.01], [0.005, 2.0]])
    alpha_2d = np.array([[1.0, 1.0], [2.0, 2.0]])
    r_2d = compute_primitive_cutoff_radius(c_2d, alpha_2d, angm=0, deriv_order=0, tol_screen=0.5)
    assert not np.any(np.isnan(r_2d))
    assert r_2d[0, 1] == 0.0
    assert r_2d[1, 0] == 0.0
    assert r_2d[0, 0] > 0.0
    assert r_2d[1, 1] > 0.0
