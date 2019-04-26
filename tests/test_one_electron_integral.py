"""Test gbasis.one_electron_integral."""
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.one_electron_integral import OneElectronCoulomb, OneElectronIntegral
import numpy as np


def test_one_electron_integral_input():
    """Test one_electron_integral.OneElectronIntegral."""
    s_type_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    s_type_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.02])
    )
    test_12 = OneElectronIntegral.construct_array_contraction(
        s_type_one, s_type_two, np.array([0.0, 0.0, 0.0]), OneElectronCoulomb.boys_func
    )
    test_21 = OneElectronIntegral.construct_array_contraction(
        s_type_two, s_type_one, np.array([0.0, 0.0, 0.0]), OneElectronCoulomb.boys_func
    )
    assert np.allclose(test_12, np.transpose(test_21, (2, 3, 0, 1)))


# TODO: Add real tests (H-H chain, etc.)
