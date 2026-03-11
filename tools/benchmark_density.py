"""
Benchmark script for density-related integrals in gbasis.

Measures execution time of:
- compute_intracule
- compute_extracule
"""

import time
import numpy as np

from gbasis.integrals.density import compute_intracule, compute_extracule


class DummyShell:
    """
    Minimal shell object for benchmarking.
    """

    def __init__(self):
        self.exps = np.array([1.0])
        self.coeffs = np.array([[1.0]])
        self.norm_prim_cart = np.array([[1.0]])

        self.angmom_components_cart = [(0, 0, 0)]
        self.angmom_components = [(0, 0, 0)]

        self.coord = np.array([0.0, 0.0, 0.0])

        self.num_seg_cont = 1
        self.num_cart = 1


def benchmark_intracule(shells):
    start = time.perf_counter()
    compute_intracule(shells)
    elapsed = time.perf_counter() - start
    print(f"intracule: {elapsed:.6f} seconds")


def benchmark_extracule(shells):
    start = time.perf_counter()
    compute_extracule(shells)
    elapsed = time.perf_counter() - start
    print(f"extracule: {elapsed:.6f} seconds")


def main(n_shells=20):
    shells = [DummyShell() for _ in range(n_shells)]

    print(f"Benchmarking with {n_shells} shells\n")

    benchmark_intracule(shells)
    benchmark_extracule(shells)


def main(n_shells=20):
    shells = [DummyShell() for _ in range(n_shells)]

    print(f"Benchmarking with {n_shells} shells\n")

    start = time.perf_counter()
    compute_intracule(shells)
    print(f"intracule: {time.perf_counter() - start:.6f} seconds")

    start = time.perf_counter()
    compute_extracule(shells)
    print(f"extracule: {time.perf_counter() - start:.6f} seconds")

if __name__ == "__main__":
    main()