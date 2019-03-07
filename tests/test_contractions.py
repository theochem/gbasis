"""Test gbasis.contractions."""
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
import pytest
from utils import skip_init


def test_charge_setter():
    """Test setter for ContractedCartesianGaussians.charge."""
    test = skip_init(ContractedCartesianGaussians)
    test.charge = 2
    assert isinstance(test._charge, float) and test._charge == 2
    test.charge = -2
    assert isinstance(test._charge, float) and test._charge == -2
    test.charge = 2.5
    assert isinstance(test._charge, float) and test._charge == 2.5
    with pytest.raises(TypeError):
        test.charge = "0"
    with pytest.raises(TypeError):
        test.charge = None


def test_charge_getter():
    """Test getter for ContractedCartesianGaussians.charge."""
    test = skip_init(ContractedCartesianGaussians)
    test._charge = 2
    assert test.charge == 2


def test_coord_setter():
    """Test setter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test.coord = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )
    test.coord = np.array([1, 2, 3])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )

    with pytest.raises(TypeError):
        test.coord = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2])
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2, 3], dtype=bool)


def test_coord_getter():
    """Test getter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test._coord = 2
    assert test.coord == 2
