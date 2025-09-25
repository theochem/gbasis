"""Test gbasis.screening"""

import numpy as np
import pytest
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.screening import is_two_index_overlap_screened, compute_primitive_upper_bound
from gbasis.screening import compute_primitive_cutoff_radius
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
        n = (
            (2 * alpha / np.pi) ** 0.25
            * (4 * alpha) ** (angm / 2)
            / np.sqrt(factorial2(2 * angm + 1))
        )
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
        n = (
            (2 * alpha / np.pi) ** 0.25
            * (4 * alpha) ** (angm / 2)
            / np.sqrt(factorial2(2 * angm + 1))
        )
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
