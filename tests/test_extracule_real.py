import numpy as np

from gbasis.parsers import parse_gbs
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals.density import compute_extracule


def convert_to_shells(parsed_shells):
    """
    Convert parsed GBS tuples into GeneralizedContractionShell objects.
    """

    origin = np.array([0.0, 0.0, 0.0], dtype=float)

    shells = []

    for angmom, exps, coeffs in parsed_shells:

        shell = GeneralizedContractionShell(
            angmom,  # angular momentum
            origin,  # center coordinate
            np.array(coeffs, dtype=float),  # contraction coefficients
            np.array(exps, dtype=float),  # exponents
            "cartesian",  # coordinate type
        )

        shells.append(shell)

    return shells


def test_extracule_real_basis():
    """
    Validate compute_extracule using real GBasis basis data.
    """

    basis_dict = parse_gbs("tests/data_631g.gbs")

    parsed_shells = basis_dict["H"]

    shells = convert_to_shells(parsed_shells)

    result = compute_extracule(shells)

    # shape validation
    assert result.shape == (len(shells), len(shells))

    # symmetry validation
    assert np.allclose(result, result.T)

    # diagonal must be non-negative
    assert np.all(result.diagonal() >= 0)
