"""Test gbasis.base.BaseGuassianRelatedArray."""
from gbasis.base import BaseGaussianRelatedArray
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
import pytest
from utils import disable_abstract, skip_init


def test_init():
    """Test base.BaseGaussianRelatedArray."""
    Test = disable_abstract(BaseGaussianRelatedArray)  # noqa: N806
    test = skip_init(Test)
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    assert not hasattr(test, "_axes_contractions")
    with pytest.raises(TypeError):
        Test.__init__(test, set([contractions]))
    with pytest.raises(ValueError):
        Test.__init__(test, [])
    with pytest.raises(TypeError):
        Test.__init__(test, [1])
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions.__dict__])

    Test.__init__(test, [contractions])
    assert test._axes_contractions == ((contractions,),)
    Test.__init__(test, [contractions, contractions])
    assert test._axes_contractions == ((contractions, contractions),)
    Test.__init__(test, [contractions, contractions], [contractions])
    assert test._axes_contractions == ((contractions, contractions), (contractions,))


def test_contruct_array_contraction():
    """Test base.BaseGaussianRelatedArray.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(  # noqa: N806
        BaseGaussianRelatedArray,
        dict_overwrite={
            "construct_array_contraction": BaseGaussianRelatedArray.construct_array_contraction
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_cartesian():
    """Test base.BaseGaussianRelatedArray.construct_array_cartesian."""
    # enable only the abstract method construct_array_cartesian
    Test = disable_abstract(  # noqa: N806
        BaseGaussianRelatedArray,
        dict_overwrite={
            "construct_array_cartesian": BaseGaussianRelatedArray.construct_array_cartesian
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_spherical():
    """Test base.BaseGaussianRelatedArray.construct_array_spherical."""
    # enable only the abstract method construct_array_spherical
    Test = disable_abstract(  # noqa: N806
        BaseGaussianRelatedArray,
        dict_overwrite={
            "construct_array_spherical": BaseGaussianRelatedArray.construct_array_spherical
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_lincomb():
    """Test base.BaseGaussianRelatedArray.construct_array_lincomb."""
    # enable only the abstract method construct_array_lincomb
    Test = disable_abstract(  # noqa: N806
        BaseGaussianRelatedArray,
        dict_overwrite={
            "construct_array_lincomb": BaseGaussianRelatedArray.construct_array_lincomb
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])
