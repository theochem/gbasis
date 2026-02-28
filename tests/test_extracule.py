import numpy as np

from gbasis.integrals.density import compute_extracule


class DummyTensor:
    def __init__(self):
        self.data = np.array([1.0])
        self.nnz = 1


class DummyShell:
    pass


import gbasis.integrals.density as density

def dummy_overlap(shells):
    return DummyTensor()

density.arbitrary_order_overlap = dummy_overlap


def test_extracule_shape():

    shells = [DummyShell(), DummyShell()]

    result = compute_extracule(shells)

    assert result.shape == (2,2)