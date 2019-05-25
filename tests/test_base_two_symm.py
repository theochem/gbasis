"""Test gbasis.base_two_symm."""
from gbasis.base_two_asymm import BaseTwoIndexAsymmetric
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.spherical import generate_transformation
import numpy as np
import pytest
from utils import disable_abstract, skip_init


def test_init():
    """Test BaseTwoIndexSymmetric.__init__."""
    Test = disable_abstract(BaseTwoIndexSymmetric)  # noqa: N806
    test = skip_init(Test)
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test.__init__(test, [contractions])
    assert test._axes_contractions[0][0] == contractions
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions])


def test_contractions():
    """Test BaseTwoIndexSymmetric.constractions."""
    Test = disable_abstract(BaseTwoIndexSymmetric)  # noqa: N806
    cont = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    test = Test([cont])
    assert test.contractions[0] == cont


def test_contruct_array_contraction():
    """Test BaseTwoIndexSymmetric.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": BaseTwoIndexSymmetric.construct_array_contraction
        },
    )
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


# TODO: test only involves four blocks. This is not big enough to catch the difference between the
# ordering of the triu and tril indices. For example, triu_indices for a 3x3 matrix is 0, 1, 2, 4,
# 5, 8 and tril_indices is 0, 3, 4, 6, 7, 8. We aim to assign the triu blocks to the tril blocks
# after taking their transpose. To do so, the mapping from triu to tril indices is 0 -> 0, 1
# -> 3, 2 -> 6, 4 -> 4, 5 -> 7, and 8 -> 8. The triu blocks cannot simply assigned in the same order
# to the tril blocks because the ordering is different.
def test_contruct_array_cartesian():
    """Test BaseTwoIndexSymmetric.construct_array_cartesian."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont1, cont2, a=2: np.ones((1, 2, 1, 2)) * a
        },
    )
    contractions.norm_cont = np.ones((1, 2))
    test = Test([contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((2, 2)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((2, 2)) * 3)
    with pytest.raises(TypeError):
        test.construct_array_cartesian(bad_keyword=3)

    test = Test([contractions, contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((4, 4)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((4, 4)) * 3)

    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(2, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(cont_one.num_contr * cont_two.num_contr, dtype=float).reshape(
                    1, cont_one.num_contr, 1, cont_two.num_contr
                )
                * a
            )
        },
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_contr))
    cont_two.norm_cont = np.ones((1, cont_two.num_contr))
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_cartesian(),
        np.vstack(
            [
                np.hstack([np.arange(9).reshape(3, 3).T * 2, np.arange(18).reshape(3, 6) * 2]),
                np.hstack([np.arange(18).reshape(3, 6).T * 2, np.arange(36).reshape(6, 6).T * 2]),
            ]
        ),
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        np.vstack(
            [
                np.hstack([np.arange(9).reshape(3, 3).T * 3, np.arange(18).reshape(3, 6) * 3]),
                np.hstack([np.arange(18).reshape(3, 6).T * 3, np.arange(36).reshape(6, 6).T * 3]),
            ]
        ),
    )

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(cont_one.num_contr * cont_two.num_contr * 2, dtype=float).reshape(
                    1, cont_one.num_contr, 1, cont_two.num_contr, 2
                )
                * a
            )
        },
    )
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_cartesian(),
        np.vstack(
            [
                np.hstack(
                    [
                        np.swapaxes(np.arange(18).reshape(3, 3, 2), 0, 1) * 2,
                        np.arange(36).reshape(3, 6, 2) * 2,
                    ]
                ),
                np.hstack(
                    [
                        np.swapaxes(np.arange(36).reshape(3, 6, 2), 0, 1) * 2,
                        np.swapaxes(np.arange(72).reshape(6, 6, 2), 0, 1) * 2,
                    ]
                ),
            ]
        ),
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.swapaxes(np.arange(18).reshape(3, 3, 2), 0, 1) * 3,
                        np.arange(36).reshape(3, 6, 2) * 3,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        np.swapaxes(np.arange(36).reshape(3, 6, 2), 0, 1) * 3,
                        np.swapaxes(np.arange(72).reshape(6, 6, 2), 0, 1) * 3,
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
    )

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            # NOTE: assume that cont_one and cont_two will always be cont_one and cont_two defined
            # above
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: np.arange(
                2 * cont_one.num_contr * 2 * cont_two.num_contr, dtype=float
            ).reshape(2, cont_one.num_contr, 2, cont_two.num_contr)
            * a
        },
    )
    cont_one.norm_cont = np.ones((2, cont_one.num_contr))
    cont_two.norm_cont = np.ones((2, cont_two.num_contr))
    test = Test([cont_one, cont_two])
    matrix_11 = np.arange(2 * cont_one.num_contr * 2 * cont_one.num_contr).reshape(
        2, cont_one.num_contr, 2, cont_one.num_contr
    )
    matrix_12 = np.arange(2 * cont_one.num_contr * 2 * cont_two.num_contr).reshape(
        2, cont_one.num_contr, 2, cont_two.num_contr
    )
    matrix_22 = np.arange(2 * cont_two.num_contr * 2 * cont_two.num_contr).reshape(
        2, cont_two.num_contr, 2, cont_two.num_contr
    )
    assert np.allclose(
        test.construct_array_cartesian(),
        np.vstack(
            [
                np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack([matrix_11[0, :, 0, :], matrix_11[0, :, 1, :]]),
                                np.hstack([matrix_11[1, :, 0, :], matrix_11[1, :, 1, :]]),
                            ]
                        ).T,
                        np.vstack(
                            [
                                np.hstack([matrix_12[0, :, 0, :], matrix_12[0, :, 1, :]]),
                                np.hstack([matrix_12[1, :, 0, :], matrix_12[1, :, 1, :]]),
                            ]
                        ),
                    ]
                ),
                np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack([matrix_12[0, :, 0, :], matrix_12[0, :, 1, :]]),
                                np.hstack([matrix_12[1, :, 0, :], matrix_12[1, :, 1, :]]),
                            ]
                        ).T,
                        np.vstack(
                            [
                                np.hstack([matrix_22[0, :, 0, :], matrix_22[0, :, 1, :]]),
                                np.hstack([matrix_22[1, :, 0, :], matrix_22[1, :, 1, :]]),
                            ]
                        ).T,
                    ]
                ),
            ]
        )
        * 2,
    )


# TODO: test only involves four blocks. This is not big enough to catch the difference between the
# ordering of the triu and tril indices. For example, triu_indices for a 3x3 matrix is 0, 1, 2, 4,
# 5, 8 and tril_indices is 0, 3, 4, 6, 7, 8. We aim to assign the triu blocks to the tril blocks
# after taking their transpose. To do so, the mapping from triu to tril indices is 0 -> 0, 1
# -> 3, 2 -> 6, 4 -> 4, 5 -> 7, and 8 -> 8. The triu blocks cannot simply assigned in the same order
# to the tril blocks because the ordering is different.
def test_contruct_array_spherical():
    """Test BaseTwoIndexSymmetric.construct_array_spherical."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    transform = generate_transformation(1, contractions.angmom_components, "left")

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_spherical(),
        (transform.dot(np.arange(9).reshape(3, 3)).dot(transform.T)).T * 2,
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        (transform.dot(np.arange(9).reshape(3, 3)).dot(transform.T)).T * 3,
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical(bad_keyword=3)

    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(2, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    transform_one = generate_transformation(1, cont_one.angmom_components, "left")
    transform_two = generate_transformation(2, cont_two.angmom_components, "left")

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(cont_one.num_contr * cont_two.num_contr * 2, dtype=float).reshape(
                    1, cont_one.num_contr, 1, cont_two.num_contr, 2
                )
                * a
            )
        },
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_contr))
    cont_two.norm_cont = np.ones((1, cont_two.num_contr))
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_spherical(a=4),
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.tensordot(
                            transform_one,
                            np.tensordot(transform_one, np.arange(18).reshape(3, 3, 2), (1, 0)),
                            (1, 1),
                        )
                        * 4,
                        np.swapaxes(
                            np.tensordot(
                                transform_two,
                                np.tensordot(transform_one, np.arange(36).reshape(3, 6, 2), (1, 0)),
                                (1, 1),
                            ),
                            0,
                            1,
                        )
                        * 4,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        np.tensordot(
                            transform_two,
                            np.tensordot(transform_one, np.arange(36).reshape(3, 6, 2), (1, 0)),
                            (1, 1),
                        )
                        * 4,
                        np.tensordot(
                            transform_two,
                            np.tensordot(transform_two, np.arange(72).reshape(6, 6, 2), (1, 0)),
                            (1, 1),
                        )
                        * 4,
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
    )

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(2 * cont_one.num_contr * 2 * cont_two.num_contr, dtype=float).reshape(
                    2, cont_one.num_contr, 2, cont_two.num_contr
                )
                * a
            )
        },
    )
    cont_one.norm_cont = np.ones((2, cont_one.num_contr))
    cont_two.norm_cont = np.ones((2, cont_two.num_contr))
    test = Test([cont_one, cont_two])

    matrix_11 = np.arange(2 * cont_one.num_contr * 2 * cont_one.num_contr).reshape(
        2, cont_one.num_contr, 2, cont_one.num_contr
    )
    matrix_11 = np.swapaxes(np.tensordot(transform_one, matrix_11, (1, 1)), 0, 1)
    matrix_11 = np.moveaxis(np.tensordot(transform_one, matrix_11, (1, 3)), 0, 3)
    matrix_12 = np.arange(2 * cont_one.num_contr * 2 * cont_two.num_contr).reshape(
        2, cont_one.num_contr, 2, cont_two.num_contr
    )
    matrix_12 = np.swapaxes(np.tensordot(transform_one, matrix_12, (1, 1)), 0, 1)
    matrix_12 = np.moveaxis(np.tensordot(transform_two, matrix_12, (1, 3)), 0, 3)
    matrix_22 = np.arange(2 * cont_two.num_contr * 2 * cont_two.num_contr).reshape(
        2, cont_two.num_contr, 2, cont_two.num_contr
    )
    matrix_22 = np.swapaxes(np.tensordot(transform_two, matrix_22, (1, 1)), 0, 1)
    matrix_22 = np.moveaxis(np.tensordot(transform_two, matrix_22, (1, 3)), 0, 3)
    assert np.allclose(
        test.construct_array_spherical(),
        np.vstack(
            [
                np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack([matrix_11[0, :, 0, :], matrix_11[0, :, 1, :]]),
                                np.hstack([matrix_11[1, :, 0, :], matrix_11[1, :, 1, :]]),
                            ]
                        ).T,
                        np.vstack(
                            [
                                np.hstack([matrix_12[0, :, 0, :], matrix_12[0, :, 1, :]]),
                                np.hstack([matrix_12[1, :, 0, :], matrix_12[1, :, 1, :]]),
                            ]
                        ),
                    ]
                ),
                np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack([matrix_12[0, :, 0, :], matrix_12[0, :, 1, :]]),
                                np.hstack([matrix_12[1, :, 0, :], matrix_12[1, :, 1, :]]),
                            ]
                        ).T,
                        np.vstack(
                            [
                                np.hstack([matrix_22[0, :, 0, :], matrix_22[0, :, 1, :]]),
                                np.hstack([matrix_22[1, :, 0, :], matrix_22[1, :, 1, :]]),
                            ]
                        ).T,
                    ]
                ),
            ]
        )
        * 2,
    )


def test_contruct_array_mix():
    """Test BaseTwoIndex.construct_array_mix."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    test = Test([contractions])
    assert np.allclose(test.construct_array_spherical(), test.construct_array_mix(["spherical"]))
    assert np.allclose(
        test.construct_array_spherical(a=3), test.construct_array_mix(["spherical"], a=3)
    )
    assert np.allclose(test.construct_array_cartesian(), test.construct_array_mix(["cartesian"]))
    assert np.allclose(
        test.construct_array_cartesian(a=3), test.construct_array_mix(["cartesian"], a=3)
    )

    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(2, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(cont_one.num_contr * cont_two.num_contr * 2, dtype=float).reshape(
                    1, cont_one.num_contr, 1, cont_two.num_contr, 2
                )
                * a
            )
        },
    )
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2)
    )
    assert np.allclose(
        test.construct_array_spherical(a=3), test.construct_array_mix(["spherical"] * 2, a=3)
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2)
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3), test.construct_array_mix(["cartesian"] * 2, a=3)
    )

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(2 * cont_one.num_contr * 2 * cont_two.num_contr, dtype=float).reshape(
                    2, cont_one.num_contr, 2, cont_two.num_contr
                )
                * a
            )
        },
    )
    cont_one.norm_cont = np.ones((2, cont_one.num_contr))
    cont_two.norm_cont = np.ones((2, cont_two.num_contr))
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2)
    )
    assert np.allclose(
        test.construct_array_spherical(a=3), test.construct_array_mix(["spherical"] * 2, a=3)
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2)
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3), test.construct_array_mix(["cartesian"] * 2, a=3)
    )

    # check coord_types type
    with pytest.raises(TypeError):
        test.construct_array_mix(np.array(["cartesian"] * 2), a=3),
    # check coord_types content
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian", "something"], a=3),
    # check coord_types length
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian"] * 3, a=3),


def test_contruct_array_lincomb():
    """Test BaseTwoIndexSymmetric.construct_array_lincomb."""
    contractions = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    sph_transform = generate_transformation(1, contractions.angmom_components, "left")
    orb_transform = np.random.rand(3, 3)

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, "cartesian"),
        (orb_transform.dot(np.arange(9).reshape(3, 3)).dot(orb_transform.T).T * 2),
    )
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, "spherical"),
        (
            orb_transform.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform.T)
            .T
            * 2
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, "spherical", a=3),
        (
            orb_transform.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform.T)
            .T
            * 3
        ),
    )
    with pytest.raises(TypeError):
        test.construct_array_lincomb(orb_transform, "spherical", bad_keyword=3)
    with pytest.raises(TypeError):
        test.construct_array_lincomb(orb_transform, "bad", keyword=3)

    Test = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: (
                np.arange(cont_one.num_contr * cont_two.num_contr * 2, dtype=float).reshape(
                    1, cont_one.num_contr, 1, cont_two.num_contr, 2
                )
                * a
            )
        },
    )
    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(2, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_one.norm_cont = np.ones((1, cont_one.num_contr))
    cont_two.norm_cont = np.ones((1, cont_two.num_contr))
    test = Test([cont_one, cont_two])
    sph_transform_one = generate_transformation(1, cont_one.angmom_components, "left")
    sph_transform_two = generate_transformation(2, cont_two.angmom_components, "left")
    orb_transform = np.random.rand(8, 8)
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, "spherical", a=4),
        np.swapaxes(
            np.tensordot(
                orb_transform,
                np.tensordot(
                    orb_transform,
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.tensordot(
                                        sph_transform_one,
                                        np.tensordot(
                                            sph_transform_one,
                                            np.arange(18).reshape(3, 3, 2),
                                            (1, 0),
                                        ),
                                        (1, 1),
                                    )
                                    * 4,
                                    np.swapaxes(
                                        np.tensordot(
                                            sph_transform_two,
                                            np.tensordot(
                                                sph_transform_one,
                                                np.arange(36).reshape(3, 6, 2),
                                                (1, 0),
                                            ),
                                            (1, 1),
                                        ),
                                        0,
                                        1,
                                    )
                                    * 4,
                                ],
                                axis=1,
                            ),
                            np.concatenate(
                                [
                                    np.tensordot(
                                        sph_transform_two,
                                        np.tensordot(
                                            sph_transform_one,
                                            np.arange(36).reshape(3, 6, 2),
                                            (1, 0),
                                        ),
                                        (1, 1),
                                    )
                                    * 4,
                                    np.tensordot(
                                        sph_transform_two,
                                        np.tensordot(
                                            sph_transform_two,
                                            np.arange(72).reshape(6, 6, 2),
                                            (1, 0),
                                        ),
                                        (1, 1),
                                    )
                                    * 4,
                                ],
                                axis=1,
                            ),
                        ],
                        axis=0,
                    ),
                    (1, 0),
                ),
                (1, 1),
            ),
            0,
            1,
        ),
    )
    orb_transform = np.random.rand(9, 9)
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, ("spherical", "cartesian"), a=4),
        np.swapaxes(
            np.tensordot(
                orb_transform,
                np.tensordot(
                    orb_transform,
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.tensordot(
                                        sph_transform_one,
                                        np.tensordot(
                                            sph_transform_one,
                                            np.arange(18).reshape(3, 3, 2),
                                            (1, 0),
                                        ),
                                        (1, 1),
                                    )
                                    * 4,
                                    np.tensordot(
                                        sph_transform_one, np.arange(36).reshape(3, 6, 2), (1, 0)
                                    )
                                    * 4,
                                ],
                                axis=1,
                            ),
                            np.concatenate(
                                [
                                    np.swapaxes(
                                        np.tensordot(
                                            sph_transform_one,
                                            np.arange(36).reshape(3, 6, 2),
                                            (1, 0),
                                        ),
                                        0,
                                        1,
                                    )
                                    * 4,
                                    np.swapaxes(np.arange(72).reshape(6, 6, 2), 0, 1) * 4,
                                ],
                                axis=1,
                            ),
                        ],
                        axis=0,
                    ),
                    (1, 0),
                ),
                (1, 1),
            ),
            0,
            1,
        ),
    )


def test_compare_two_asymm():
    """Test BaseTwoIndexSymmetric by comparing it against BaseTwoIndexAsymmetric."""
    cont_one = ContractedCartesianGaussians(1, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    cont_two = ContractedCartesianGaussians(2, np.array([1, 2, 3]), 0, np.ones(1), np.ones(1))
    sph_orb_transform = np.random.rand(8, 8)
    cart_orb_transform = np.random.rand(9, 9)

    def construct_array_contraction(self, cont_one, cont_two, a=2):
        """Temporary symmetric function for testing."""
        one_size = cont_one.num_contr
        two_size = cont_two.num_contr
        output = (
            np.arange(one_size)[None, :, None, None, None]
            + np.arange(two_size)[None, None, None, :, None]
            + np.arange(5, 10)[None, None, None, None, :]
        ).astype(float)
        return output * a

    TestSymmetric = disable_abstract(  # noqa: N806
        BaseTwoIndexSymmetric,
        dict_overwrite={"construct_array_contraction": construct_array_contraction},
    )
    TestAsymmetric = disable_abstract(  # noqa: N806
        BaseTwoIndexAsymmetric,
        dict_overwrite={"construct_array_contraction": construct_array_contraction},
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_contr))
    cont_two.norm_cont = np.ones((1, cont_two.num_contr))

    test_symm = TestSymmetric([cont_one, cont_two])
    test_asymm = TestAsymmetric([cont_one, cont_two], [cont_one, cont_two])

    assert np.allclose(
        test_symm.construct_array_contraction(cont_one, cont_one),
        test_asymm.construct_array_contraction(cont_one, cont_one),
    )
    assert np.allclose(
        test_symm.construct_array_contraction(cont_one, cont_two),
        test_asymm.construct_array_contraction(cont_one, cont_two),
    )
    assert np.allclose(
        test_symm.construct_array_contraction(cont_two, cont_one),
        test_asymm.construct_array_contraction(cont_two, cont_one),
    )
    assert np.allclose(
        test_symm.construct_array_contraction(cont_two, cont_two),
        test_asymm.construct_array_contraction(cont_two, cont_two),
    )
    assert np.allclose(
        test_symm.construct_array_cartesian(), test_asymm.construct_array_cartesian()
    )
    assert np.allclose(
        test_symm.construct_array_spherical(), test_asymm.construct_array_spherical()
    )
    assert np.allclose(
        test_symm.construct_array_lincomb(sph_orb_transform, "spherical"),
        test_asymm.construct_array_lincomb(
            sph_orb_transform, sph_orb_transform, "spherical", "spherical"
        ),
    )
    assert np.allclose(
        test_symm.construct_array_lincomb(cart_orb_transform, "cartesian"),
        test_asymm.construct_array_lincomb(
            cart_orb_transform, cart_orb_transform, "cartesian", "cartesian"
        ),
    )
