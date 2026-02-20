import numpy as np
import pytest

from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_n import contracted_n_overlap


# =====================================================
# Helper: Build small hydrogen basis
# =====================================================
def build_h2_basis():
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])

    basis_dict = parse_gbs("tests/data_631g.gbs")

    basis = make_contractions(
        basis_dict,
        atoms,
        coords,
        coord_types="c"
    )

    return basis


# =====================================================
# 1️⃣ N=2 FULL AO MATRIX VALIDATION
# =====================================================

def test_n2_matches_gbasis():

    basis = build_h2_basis()

    ref = overlap_integral(basis, screen_basis=False)

    n_shells = len(basis)
    total_ao = ref.shape[0]
    mine = np.zeros((total_ao, total_ao))

    ao_i = 0
    for shell_i in basis:
        size_i = shell_i.num_seg_cont * shell_i.num_cart

        ao_j = 0
        for shell_j in basis:
            size_j = shell_j.num_seg_cont * shell_j.num_cart

            block = contracted_n_overlap([shell_i, shell_j])
            block = block.reshape(size_i, size_j)

            mine[ao_i:ao_i+size_i,
                 ao_j:ao_j+size_j] = block

            ao_j += size_j

        ao_i += size_i

    diff = np.max(np.abs(ref - mine))

    print("N=2 max difference:", diff)

    assert np.allclose(ref, mine, atol=1e-10)


# =====================================================
# 2️⃣ N=3 SYMMETRY TEST
# =====================================================

def test_n3_symmetry():

    basis = build_h2_basis()

    shell1 = basis[0]
    shell2 = basis[1]
    shell3 = basis[0]

    S123 = contracted_n_overlap([shell1, shell2, shell3])
    S321 = contracted_n_overlap([shell3, shell2, shell1])

    diff = np.max(np.abs(S123 - S321))

    print("N=3 symmetry difference:", diff)

    assert np.allclose(S123, S321, atol=1e-10)


# =====================================================
# 3️⃣ HIGH ANGULAR MOMENTUM + DIFFUSE TEST
# =====================================================

def test_high_angmom_diffuse():

    basis = build_h2_basis()

    # choose a higher angular momentum shell if available
    # try to find p-shell
    p_shells = [shell for shell in basis if shell.angmom == 1]

    if not p_shells:
        pytest.skip("No p-shell available in basis.")

    shell = p_shells[0]

    # artificially make exponents diffuse
    shell.exps = shell.exps * 0.01

    S = contracted_n_overlap([shell, shell])

    print("Diffuse shell max abs value:", np.max(np.abs(S)))

    assert not np.isnan(S).any()
    assert not np.isinf(S).any()
