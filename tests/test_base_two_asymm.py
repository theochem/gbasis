"""Test gbasis.base_two_asymm."""
from gbasis.base_two_asymm import BaseTwoIndexAsymmetric
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.spherical import generate_transformation
import numpy as np
import pytest
from utils import disable_abstract, skip_init


def test_init():
    """Test BaseTwoIndexAsymmetric.__init__."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)  # noqa: N806
    test = skip_init(Test)
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test.__init__(test, [contractions], [contractions])
    assert test._axes_contractions == ((contractions,), (contractions,))
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions])
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions], [contractions])


def test_contractions_one():
    """Test BaseTwoIndexAsymmetric.constractions_one."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)  # noqa: N806
    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    test = Test([cont_one], [cont_two])
    assert test.contractions_one[0] == cont_one


def test_contractions_two():
    """Test BaseTwoIndexAsymmetric.constractions_two."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)  # noqa: N806
    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    test = Test([cont_one], [cont_two])
    assert test.contractions_two[0] == cont_two


def test_contruct_array_contraction():
    """Test BaseTwoIndexAsymmetric.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": BaseTwoIndexAsymmetric.construct_array_contraction
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_cartesian():
    """Test BaseTwoIndexAsymmetric.construct_array_cartesian."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont1, cont2, a=2: np.ones((1, 2, 1, 2)) * a
        },
    )
    test = Test([contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((2, 2)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((2, 2)) * 3)
    with pytest.raises(TypeError):
        test.construct_array_cartesian(bad_keyword=3)

    test = Test([contractions, contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((4, 2)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((4, 2)) * 3)

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.ones((1, 2, 1, 5)) * a
            )
        },
    )
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((4, 5)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((4, 5)) * 3)

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.ones((2, 2, 2, 5)) * a
            )
        },
    )
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((8, 10)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((8, 10)) * 3)


def test_contruct_array_spherical():
    """Test BaseTwoIndexAsymmetric.construct_array_spherical."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    transform = generate_transformation(1, contractions.angmom_components, "left")

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9).reshape(1, 3, 1, 3) * a
            )
        },
    )
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        transform.dot(np.arange(9).reshape(3, 3)).dot(transform.T) * 2,
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        transform.dot(np.arange(9).reshape(3, 3)).dot(transform.T) * 3,
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical(bad_keyword=3)

    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack([transform.dot(np.arange(9).reshape(3, 3).dot(transform.T)) * 2] * 2),
    )

    matrix = np.arange(36).reshape(2, 3, 2, 3)
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: matrix * a
        },
    )
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack(
            [
                np.hstack(
                    [
                        transform.dot(matrix[0, :, 0, :]).dot(transform.T),
                        transform.dot(matrix[0, :, 1, :]).dot(transform.T),
                    ]
                ),
                np.hstack(
                    [
                        transform.dot(matrix[1, :, 0, :]).dot(transform.T),
                        transform.dot(matrix[1, :, 1, :]).dot(transform.T),
                    ]
                ),
            ]
        )
        * 2,
    )
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack(
            [
                np.hstack(
                    [
                        transform.dot(matrix[0, :, 0, :]).dot(transform.T),
                        transform.dot(matrix[0, :, 1, :]).dot(transform.T),
                    ]
                ),
                np.hstack(
                    [
                        transform.dot(matrix[1, :, 0, :]).dot(transform.T),
                        transform.dot(matrix[1, :, 1, :]).dot(transform.T),
                    ]
                ),
            ]
            * 2
        )
        * 2,
    )


def test_contruct_array_spherical_lincomb():
    """Test BaseTwoIndexAsymmetric.construct_array_spherical_lincomb."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    sph_transform = generate_transformation(1, contractions.angmom_components, "left")
    orb_transform_one = np.random.rand(3, 3)
    orb_transform_two = np.random.rand(3, 3)

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9).reshape(1, 3, 1, 3) * a
            )
        },
    )
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform_one, orb_transform_two),
        (
            orb_transform_one.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform_two.T)
            * 2
        ),
    )
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform_one, orb_transform_two, a=3),
        (
            orb_transform_one.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform_two.T)
            * 3
        ),
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical_lincomb(bad_keyword=3)

    orb_transform_one = np.random.rand(3, 6)
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical_lincomb(orb_transform_one, orb_transform_two),
        orb_transform_one.dot(
            np.vstack(
                [sph_transform.dot(np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2] * 2
            ).dot(orb_transform_two.T)
        ),
    )
