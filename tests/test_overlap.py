"""Test gbasis.overlap."""
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.moment_int import _compute_multipole_moment_integrals
from gbasis.overlap import Overlap
import numpy as np
import pytest
from scipy.special import factorial2


def test_overlap_construct_array_contraction():
    """Test gbasis.overlap.Overlap.construct_array_contraction."""
    test_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    test_two = ContractedCartesianGaussians(
        2, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.2, 0.02])
    )
    answer = np.array(
        [
            [
                _compute_multipole_moment_integrals(
                    np.array([0, 0, 0]),
                    np.array([[0, 0, 0]]),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components
            ]
            for angmom_comp_one in test_one.angmom_components
        ]
    )
    assert np.allclose(
        np.squeeze(Overlap.construct_array_contraction(test_one, test_two)), np.squeeze(answer)
    )

    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(None, test_two)
