"""Test gbasis.screening"""
import numpy as np
import pytest
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.screening import is_two_index_overlap_screened
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

    print(f"Testing with bond length: {bond_length} and screen tolerance: {tol_screen}")
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
