import numpy as np
import pytest

from gbasis.integrals.overlap import Overlap
from gbasis.integrals.overlap_n import contracted_n_overlap, PrimitiveNEngine
from gbasis.contractions import GeneralizedContractionShell


def build_test_shell(center):
    ...
    return GeneralizedContractionShell(...)


def test_two_center_matches_existing_overlap():
    ...
    assert ...


def test_two_center_ss_analytic():
    ...
    assert ...


def test_primitive_px_py_pz():
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.3, 0.1, -0.2])

    alpha = 0.5

    print("px-px:",
          PrimitiveNEngine.primitive_overlap(
              [alpha, alpha],
              [A, B],
              [(1,0,0), (1,0,0)]
          ))

    print("py-py:",
          PrimitiveNEngine.primitive_overlap(
              [alpha, alpha],
              [A, B],
              [(0,1,0), (0,1,0)]
          ))

    print("pz-pz:",
          PrimitiveNEngine.primitive_overlap(
              [alpha, alpha],
              [A, B],
              [(0,0,1), (0,0,1)]
          ))

    assert True  # temporary so pytest runs it
