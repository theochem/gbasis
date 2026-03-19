import numpy as np

from gbasis.integrals.density import (
    compute_intracule,
    compute_extracule,
)


class DummyShell:

    def __init__(self):

        self.exps = np.array([1.0])
        self.coeffs = np.array([[1.0]])
        self.norm_prim_cart = np.array([[1.0]])
        self.angmom_components_cart = [(0, 0, 0)]
        self.coord = np.array([0.0, 0.0, 0.0])
        self.num_seg_cont = 1
        self.num_cart = 1


def test_compute_intracule():

    shells = [DummyShell(), DummyShell()]

    result = compute_intracule(shells)

    assert result is not None


def test_compute_extracule():

    shells = [DummyShell(), DummyShell()]

    result = compute_extracule(shells)

    assert result is not None
