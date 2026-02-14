import numpy as np
from pathlib import Path

from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap_n import build_sparse_n_overlap_tensor


def build_h2_basis():
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])

    test_dir = Path(__file__).parent
    basis_path = test_dir / "data" / "hydrogen_def2-svp.1.gbs"

    basis_dict = parse_gbs(str(basis_path))
    basis = make_contractions(
        basis_dict,
        atoms,
        coords,
        coord_types="c"
    )
    return basis


def test_sparse_builder_runs():

    basis = build_h2_basis()

    sparse_mat = build_sparse_n_overlap_tensor(basis)

    print("Sparse tensor shape:", sparse_mat.shape)
    print("Number of nonzeros:", sparse_mat.nnz)

    assert sparse_mat.shape[0] > 0
    assert sparse_mat.nnz > 0
