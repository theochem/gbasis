"""Test gbasis.contractions."""
from gbasis.contractions import ContractedCartesianGaussians
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
