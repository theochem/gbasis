import numpy as np
from gbasis.integrals.multi_overlap import three_overlap_tensor


def test_tensor():
    alpha = 1.0
    basis = [
        (alpha, np.array([0.0, 0.0, 0.0])),
        (alpha, np.array([0.0, 0.0, 1.0])),
        (alpha, np.array([0.0, 1.0, 0.0])),
    ]

    T = three_overlap_tensor(basis)

    print("Tensor shape:", T.shape)
    print("T[0,0,0] =", T[0,0,0])
    print("T[0,1,2] =", T[0,1,2])


if __name__ == "__main__":
    test_tensor()
