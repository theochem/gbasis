# Testing for 1-2Gaussian overlapimport pytest

def test_overlap():
    assert 1 + 1 == 2

import numpy as np

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals.overlap import overlap_integral


# -------------------------
# Define two 1s Gaussian shells
# -------------------------
angmom = 0  # s orbital

exps = np.array([0.5])
coeffs = np.array([[1.0]])

g1 = GeneralizedContractionShell(
    angmom=angmom,
    coord=np.array([0.0, 0.0, 0.0]),
    exps=exps,            
    coeffs=coeffs,          
    coord_type="cartesian",
)

g2 = GeneralizedContractionShell(
    angmom=angmom,
    coord=np.array([0.2, 0.0, 0.0]),
    exps=exps,
    coeffs=coeffs,
    coord_type="cartesian",
)

# -------------------------
# BASIS = list of shells
# -------------------------
basis = [g1, g2]


# Overlap matrix

S = overlap_integral(basis)

print("Overlap matrix:")
print(S)
