import numpy as np
from gbasis.parsers import parse_gbs
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals.density import compute_intracule


def convert_to_shells(parsed_shells):

    origin = np.array([0.0, 0.0, 0.0], dtype=float)

    shells = []

    for angmom, exps, coeffs in parsed_shells:

        shell = GeneralizedContractionShell(
            angmom,                              # 1
            origin,                              # 2
            np.array(coeffs, dtype=float),      # 3
            np.array(exps, dtype=float),        # 4
            "cartesian"                         # 5
        )

        shells.append(shell)

    return shells


def test_intracule_real_basis():

    basis_dict = parse_gbs("tests/data_631g.gbs")

    parsed_shells = basis_dict["H"]

    shells = convert_to_shells(parsed_shells)

    result = compute_intracule(shells)

    assert result.shape == (len(shells), len(shells))

    assert np.allclose(result, result.T)

    assert np.all(result.diagonal() >= 0)
