import time

from gbasis.integrals.density import compute_intracule


class DummyTensor:
    def __init__(self):
        self.data = [1.0]
        self.nnz = 1


class DummyShell:
    pass


import gbasis.integrals.density as density

<<<<<<< HEAD

def dummy_overlap(shells):
    return DummyTensor()


=======
def dummy_overlap(shells):
    return DummyTensor()

>>>>>>> 30adfdd (Week-5: Add tests, benchmarks, and real-basis validation for density API)
density.arbitrary_order_overlap = dummy_overlap


def test_intracule_speed():

    shells = [DummyShell() for _ in range(20)]

    start = time.time()

    compute_intracule(shells)

    elapsed = time.time() - start

<<<<<<< HEAD
    assert elapsed < 5
=======
    assert elapsed < 5
>>>>>>> 30adfdd (Week-5: Add tests, benchmarks, and real-basis validation for density API)
