import numpy as np

from gbasis.integrals.density import compute_intracule


class DummyTensor:
    def __init__(self):
        self.data = np.array([1.0])
        self.nnz = 1


class DummyShell:
    pass


# Monkeypatch arbitrary_order_overlap
import gbasis.integrals.density as density

def dummy_overlap(shells):
    return DummyTensor()

density.arbitrary_order_overlap = dummy_overlap


def test_intracule_shape():

    shells = [DummyShell(), DummyShell(), DummyShell()]

    result = compute_intracule(shells)

    assert result.shape == (3,3)


def test_intracule_empty():

    shells = []

    result = compute_intracule(shells)

    assert result.shape == (0,0)


def test_intracule_symmetry():

    shells = [DummyShell(), DummyShell()]

    result = compute_intracule(shells)

    assert np.allclose(result, result.T)
