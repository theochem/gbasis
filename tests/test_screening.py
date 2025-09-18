"""Test gbasis.screening"""
import numpy as np
import pytest
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.screening import is_two_index_overlap_screened
from utils import find_datafile
from gbasis.utils import factorial2
from gbasis.screening import compute_primitive_cutoff_radius


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


@pytest.mark.parametrize("angm", [0, 1, 2, 3])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("coeff", [0.1, 0.4, 0.8])
@pytest.mark.parametrize("tol_screen", [1e-4, 1e-8])
def test_compute_primitive_cutoff_radius(angm, alpha, coeff, tol_screen):
    """Test the computation of the primitive cutoff radius."""

    def compute_primitive_value(r, c, alpha, angm):
        """Compute the primitive value ate the given radius."""
        n = (
            (2 * alpha / np.pi) ** 0.25
            * (4 * alpha) ** (angm / 2)
            / np.sqrt(factorial2(2 * angm + 1))
        )
        return c * n * np.exp(-alpha * r**2)

    cutoff_r = compute_primitive_cutoff_radius(coeff, alpha, angm, tol_screen)
    val_over_cutoff = compute_primitive_value(cutoff_r + 1e-3, coeff, alpha, angm)

    assert (
        val_over_cutoff < tol_screen
    ), f"Value {val_over_cutoff} at r={cutoff_r + 1e-3} is not below the tolerance {tol_screen}"
