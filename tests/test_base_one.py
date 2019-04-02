"""Test gbasis.base_one."""
from gbasis.base_one import BaseOneIndex
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.spherical import generate_transformation
import numpy as np
import pytest
from utils import disable_abstract, skip_init


def test_init():
    """Test BaseOneIndex.__init__."""
    Test = disable_abstract(BaseOneIndex)  # noqa: N806
    test = skip_init(Test)
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test.__init__(test, [contractions])
    assert test._axes_contractions == ((contractions,),)
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions])


def test_contractions():
    """Test BaseOneIndex.constractions."""
    Test = disable_abstract(BaseOneIndex)  # noqa: N806
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    test = Test([contractions])
    assert test.contractions[0] == contractions


def test_contruct_array_contraction():
    """Test BaseOneIndex.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={"construct_array_contraction": BaseOneIndex.construct_array_contraction},
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_cartesian():
    """Test BaseOneIndex.construct_array_cartesian."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={"construct_array_contraction": lambda self, cont, a=2: np.ones((1, 5)) * a},
    )
    test = Test([contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones(5) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones(5) * 3)
    with pytest.raises(TypeError):
        test.construct_array_cartesian(bad_keyword=3)

    test = Test([contractions, contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones(10) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones(10) * 3)

    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={"construct_array_contraction": lambda self, cont, a=2: np.ones((2, 5)) * a},
    )
    test = Test([contractions, contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones(20) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones(20) * 3)

    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont, a=2: np.ones((2, 5, 4)) * a
        },
    )
    test = Test([contractions, contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((20, 4)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((20, 4)) * 3)


def test_contruct_array_spherical():
    """Test BaseOneIndex.construct_array_spherical."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    transform = generate_transformation(1, contractions.angmom_components, "left")

    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont, a=2: np.arange(9).reshape(1, 3, 3) * a
        },
    )
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_spherical(), transform.dot(np.arange(9).reshape(3, 3)) * 2
    )
    assert np.allclose(
        test.construct_array_spherical(a=3), transform.dot(np.arange(9).reshape(3, 3)) * 3
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical(bad_keyword=3)

    test = Test([contractions, contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack([transform.dot(np.arange(9).reshape(3, 3)) * 2] * 2),
    )

    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont, a=2: np.arange(18).reshape(2, 3, 3) * a
            )
        },
    )
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack(
            [
                transform.dot(np.arange(9).reshape(3, 3)),
                transform.dot(np.arange(9, 18).reshape(3, 3)),
            ]
        )
        * 2,
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        np.vstack(
            [
                transform.dot(np.arange(9).reshape(3, 3)),
                transform.dot(np.arange(9, 18).reshape(3, 3)),
            ]
        )
        * 3,
    )

    test = Test([contractions, contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack(
            [
                transform.dot(np.arange(9).reshape(3, 3)),
                transform.dot(np.arange(9, 18).reshape(3, 3)),
            ]
            * 2
        )
        * 2,
    )


def test_contruct_array_spherical_lincomb():
    """Test BaseOneIndex.construct_array_spherical_lincomb."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    sph_transform = generate_transformation(1, contractions.angmom_components, "left")
    orb_transform = np.random.rand(3, 3)

    Test = disable_abstract(  # noqa: N806
        BaseOneIndex,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont, a=2: np.arange(9).reshape(1, 3, 3) * a
        },
    )
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform),
        orb_transform.dot(sph_transform).dot(np.arange(9).reshape(3, 3)) * 2,
    )
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform, a=3),
        orb_transform.dot(sph_transform).dot(np.arange(9).reshape(3, 3)) * 3,
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical_lincomb(bad_keyword=3)

    orb_transform = np.random.rand(3, 6)
    test = Test([contractions, contractions])
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform),
        orb_transform.dot(np.vstack([sph_transform.dot(np.arange(9).reshape(3, 3)) * 2] * 2)),
    )
