import numpy as np
import pytest

from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap_n import contracted_n_overlap


# =====================================================
# Helper: Build hydrogen basis
# =====================================================

def build_h2_basis():

    atoms = ["H", "H"]

    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    basis_dict = parse_gbs("tests/data_631g.gbs")

    basis = make_contractions(
        basis_dict,
        atoms,
        coords,
        coord_types="c"
    )

    return basis


# =====================================================
# Test arbitrary orders N=1 to N=6
# =====================================================

@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6])
def test_arbitrary_n_orders(N):

    basis = build_h2_basis()

    shells = []

    for i in range(N):
        shells.append(basis[i % len(basis)])

    result = contracted_n_overlap(shells)

    print(f"N={N}, max abs value:", np.max(np.abs(result)))

    assert result is not None

    assert not np.isnan(result).any()

    assert not np.isinf(result).any()

    assert np.all(np.isfinite(result))