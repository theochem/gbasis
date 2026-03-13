import time
import numpy as np
from gbasis.integrals.density import compute_intracule


class DummyShell:

    def __init__(self):
        self.exps = np.array([1.0])
        self.coeffs = np.array([[1.0]])
        self.norm_prim_cart = np.array([[1.0]])
        self.angmom_components_cart = [(0, 0, 0)]
        self.angmom_components = [(0, 0, 0)]
        self.coord = np.array([0.0, 0.0, 0.0])
        self.num_seg_cont = 1
        self.num_cart = 1


def benchmark_intracule(n_shells=20):

    shells = [DummyShell() for _ in range(n_shells)]

    start = time.perf_counter()
    compute_intracule(shells)
    elapsed = time.perf_counter() - start

    print(f"compute_intracule benchmark: {elapsed:.6f} seconds for {n_shells} shells")


if __name__ == "__main__":
    benchmark_intracule()