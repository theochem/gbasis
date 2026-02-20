import numpy as np
from pathlib import Path
from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap_n import build_n_overlap_tensor


def build_h2_basis():
    atoms = ["H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    basis_path = Path(
        r"C:\Users\kiran\OneDrive\ドキュメント\nnn\gbasis\notebooks\tutorial\hydrogen_def2-svp.1.gbs"
    )

    basis_dict = parse_gbs(str(basis_path))

    return make_contractions(
        basis_dict,
        atoms,
        coords,
        coord_types="c"
    )


def test_n_overlap_tensor_builder():

    basis = build_h2_basis()

    shell1 = basis[0]
    shell2 = basis[1]

    tensor = build_n_overlap_tensor([shell1, shell2])

    print("Tensor shape:", tensor.shape)
    print("Nonzeros:", tensor.nnz)

    assert tensor.shape[0] > 0
    assert tensor.nnz > 0
