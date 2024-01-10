"""Test gbasis.base_two_asymm."""
import numpy as np
import pytest
from utils import disable_abstract, skip_init

from gbasis.base_two_asymm import BaseTwoIndexAsymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.spherical import generate_transformation


def test_init():
    """Test BaseTwoIndexAsymmetric.__init__."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)
    test = skip_init(Test)
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    Test.__init__(test, [contractions], [contractions])
    assert test._axes_contractions == ((contractions,), (contractions,))
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions])
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions], [contractions])


def test_contractions_one():
    """Test BaseTwoIndexAsymmetric.constractions_one."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)
    cont_one = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    cont_two = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    test = Test([cont_one], [cont_two])
    assert test.contractions_one[0] == cont_one


def test_contractions_two():
    """Test BaseTwoIndexAsymmetric.constractions_two."""
    Test = disable_abstract(BaseTwoIndexAsymmetric)
    cont_one = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    cont_two = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    test = Test([cont_one], [cont_two])
    assert test.contractions_two[0] == cont_two


def test_contruct_array_contraction():
    """Test BaseTwoIndexAsymmetric.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": BaseTwoIndexAsymmetric.construct_array_contraction
        },
    )
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    with pytest.raises(TypeError):
        Test([contractions])


def test_contruct_array_cartesian():
    """Test BaseTwoIndexAsymmetric.construct_array_cartesian."""
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont1, cont2, a=2: np.ones((1, 2, 1, 2)) * a
        },
    )
    contractions.norm_cont = np.ones((1, 2))
    test = Test([contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((2, 2)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((2, 2)) * 3)
    with pytest.raises(TypeError):
        test.construct_array_cartesian(bad_keyword=3)

    test = Test([contractions, contractions], [contractions])
    assert np.allclose(test.construct_array_cartesian(), np.ones((4, 2)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((4, 2)) * 3)

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.ones((1, 2, 1, 5)) * a
            )
        },
    )
    cont_one = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    cont_two = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    cont_one.norm_cont = np.ones((1, 2))
    cont_two.norm_cont = np.ones((1, 5))
    test = Test([cont_one, cont_one], [cont_two])
    assert np.allclose(test.construct_array_cartesian(), np.ones((4, 5)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((4, 5)) * 3)

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.ones((2, 2, 2, 5)) * a
            )
        },
    )
    cont_one.norm_cont = np.ones((2, 2))
    cont_two.norm_cont = np.ones((2, 5))
    test = Test([cont_one, cont_one], [cont_two])
    assert np.allclose(test.construct_array_cartesian(), np.ones((8, 10)) * 2)
    assert np.allclose(test.construct_array_cartesian(a=3), np.ones((8, 10)) * 3)


def test_contruct_array_spherical():
    """Test BaseTwoIndexAsymmetric.construct_array_spherical."""
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    transform = generate_transformation(
        1, contractions.angmom_components_cart, contractions.angmom_components_sph, "left"
    )

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))

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

    matrix = np.arange(36, dtype=float).reshape(2, 3, 2, 3)
    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: matrix * a
        },
    )
    contractions.norm_cont = np.ones((2, 3))
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


def test_contruct_array_mix():
    """Test BaseTwoIndexAsymmetric.construct_array_mix."""
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"], ["spherical"])
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        test.construct_array_mix(["spherical"], ["spherical"], a=3),
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"], ["cartesian"])
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        test.construct_array_mix(["cartesian"], ["cartesian"], a=3),
    )

    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2, ["spherical"])
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        test.construct_array_mix(["spherical"] * 2, ["spherical"], a=3),
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2, ["cartesian"])
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        test.construct_array_mix(["cartesian"] * 2, ["cartesian"], a=3),
    )

    matrix = np.arange(36, dtype=float).reshape(2, 3, 2, 3)
    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, a=2: matrix * a
        },
    )
    contractions.norm_cont = np.ones((2, 3))
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"], ["spherical"])
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        test.construct_array_mix(["spherical"], ["spherical"], a=3),
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"], ["cartesian"])
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        test.construct_array_mix(["cartesian"], ["cartesian"], a=3),
    )
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2, ["spherical"])
    )
    assert np.allclose(
        test.construct_array_spherical(a=3),
        test.construct_array_mix(["spherical"] * 2, ["spherical"], a=3),
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2, ["cartesian"])
    )
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        test.construct_array_mix(["cartesian"] * 2, ["cartesian"], a=3),
    )

    # check coord_types_one type
    with pytest.raises(TypeError):
        test.construct_array_mix(np.array(["cartesian"] * 2), ["cartesian"], a=3),
    # check coord_types_two type
    with pytest.raises(TypeError):
        test.construct_array_mix(["cartesian"] * 2, np.array(["cartesian"]), a=3),
    # check coord_types_one content
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian", "something"], ["cartesian"], a=3),
    # check coord_types_one content
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian"] * 2, ["something"], a=3),
    # check coord_types_one length
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian"] * 3, ["cartesian"], a=3),
    # check coord_types_two length
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian"] * 2, ["cartesian"] * 2, a=3),


def test_construct_array_mix_with_both_cartesian_and_spherical():
    r"""Test construct_array_mix with both a P-Type Cartesian and D-Type Spherical contractions."""
    num_pts = 1
    # Define the coefficients used to seperate which contraction block it is
    coeff_p_p_type = 2
    coeff_p_d_type = 4
    coeff_d_p_type = 5
    coeff_d_d_type = 6

    def construct_array_cont(self, cont_one, cont_two):
        if cont_one.angmom == 1:
            if cont_two.angmom == 1:
                # Return array with all values of "COEFF_P_PTYPE" with right size
                output = (
                    np.ones(cont_one.num_cart * cont_two.num_cart * num_pts, dtype=float).reshape(
                        1, cont_one.num_cart, 1, cont_two.num_cart, num_pts
                    )
                    * coeff_p_p_type
                )
            elif cont_two.angmom == 2:
                # Return array with all values of "COEFF_P_D_TYPE" with right size
                output = (
                    np.ones(cont_one.num_cart * cont_two.num_cart * num_pts, dtype=float).reshape(
                        1, cont_one.num_cart, 1, cont_two.num_cart, num_pts
                    )
                    * coeff_p_d_type
                )
        if cont_one.angmom == 2:
            if cont_two.angmom == 1:
                # Return array with all values of "COEFF_P_PTYPE" with right size
                output = (
                    np.ones(cont_one.num_cart * cont_two.num_cart * num_pts, dtype=float).reshape(
                        1, cont_one.num_cart, 1, cont_two.num_cart, num_pts
                    )
                    * coeff_d_p_type
                )
            elif cont_two.angmom == 2:
                # Return array with all values of "COEFF_D_D_TYPE" with right size
                output = (
                    np.ones(cont_one.num_cart * cont_two.num_cart * num_pts, dtype=float).reshape(
                        1, cont_one.num_cart, 1, cont_two.num_cart, num_pts
                    )
                    * coeff_d_d_type
                )
        return output

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={"construct_array_contraction": construct_array_cont},
    )
    cont_one = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    cont_two = GeneralizedContractionShell(2, np.array([1, 2, 3]), np.ones(1), np.ones(1))

    # Remove the dependence on norm constants.
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))
    test = Test([cont_one, cont_two], [cont_one, cont_two])

    # Should have shape (3 + 5, 3 + 5, NUM_PTS), due to the following:
    #           3-> Number of P-type, 5->Number of Spherical D-type.
    actual = test.construct_array_mix(["cartesian", "spherical"], ["cartesian", "spherical"])[
        :, :, 0
    ]

    # Test P-type to P-type
    assert np.allclose(actual[:3, :3], np.ones((3, 3)) * coeff_p_p_type)
    # Test P-type to D-type
    # Transformation matrix from  normalized Cartesian to normalized Spherical,
    #       Transfers [xx, xy, xz, yy, yz, zz] to [S_{22}, S_{21}, C_{20}, C_{21}, C_{22}]
    #       Obtained form iodata website or can find it in Helgeker's book.
    generate_transformation_array = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [-0.5, 0, 0, -0.5, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [np.sqrt(3.0) / 2.0, 0, 0, -np.sqrt(3.0) / 2.0, 0, 0],
        ]
    )
    assert np.allclose(
        actual[:3, 3:], np.ones((3, 6)).dot(generate_transformation_array.T) * coeff_p_d_type
    )
    assert np.allclose(
        actual[3:, :3], generate_transformation_array.dot(np.ones((6, 3))) * coeff_d_p_type
    )
    # Test D-type to D-type.
    assert np.allclose(
        actual[3:, 3:],
        generate_transformation_array.dot(np.ones(6 * 6, dtype=float).reshape(6, 6) * 6).dot(
            generate_transformation_array.T
        ),
    )


def test_contruct_array_lincomb():
    """Test BaseTwoIndexAsymmetric.construct_array_lincomb."""
    contractions = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1))
    sph_transform = generate_transformation(
        1, contractions.angmom_components_cart, contractions.angmom_components_sph, "left"
    )
    orb_transform_one = np.random.rand(3, 3)
    orb_transform_two = np.random.rand(3, 3)

    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, a=2: np.arange(9, dtype=float).reshape(1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions], [contractions])
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "cartesian", "cartesian"
        ),
        orb_transform_one.dot(np.arange(9).reshape(3, 3)).dot(orb_transform_two.T) * 2,
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "spherical", "spherical"
        ),
        (
            orb_transform_one.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform_two.T)
            * 2
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "cartesian", "spherical"
        ),
        (
            orb_transform_one.dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform_two.T)
            * 2
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "spherical", "cartesian"
        ),
        (
            orb_transform_one.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(orb_transform_two.T)
            * 2
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "spherical", "spherical", a=3
        ),
        (
            orb_transform_one.dot(sph_transform)
            .dot(np.arange(9).reshape(3, 3))
            .dot(sph_transform.T)
            .dot(orb_transform_two.T)
            * 3
        ),
    )
    with pytest.raises(TypeError):
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "spherical", "spherical", bad_keyword=3
        )
    with pytest.raises(TypeError):
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "bad", "spherical", keyword=3
        )
    with pytest.raises(TypeError):
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "cartesian", "bad", keyword=3
        )

    orb_transform_one = np.random.rand(3, 6)
    orb_transform_two = np.random.rand(3, 3)
    test = Test([contractions, contractions], [contractions])
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "spherical", "spherical"
        ),
        orb_transform_one.dot(
            np.vstack(
                [sph_transform.dot(np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2] * 2
            ).dot(orb_transform_two.T)
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, ["spherical", "cartesian"], "spherical"
        ),
        orb_transform_one.dot(
            np.vstack(
                [
                    sph_transform.dot(np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2,
                    (np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2,
                ]
            ).dot(orb_transform_two.T)
        ),
    )
    orb_transform_one = np.random.rand(3, 3)
    orb_transform_two = np.random.rand(3, 6)
    test = Test([contractions], [contractions, contractions])
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, orb_transform_two, "cartesian", ["spherical", "cartesian"]
        ),
        orb_transform_one.dot(
            np.hstack(
                [
                    (np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2,
                    np.arange(9).reshape(3, 3) * 2,
                ]
            ).dot(orb_transform_two.T)
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            None, orb_transform_two, "cartesian", ["spherical", "cartesian"]
        ),
        np.hstack(
            [(np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2, np.arange(9).reshape(3, 3) * 2]
        ).dot(orb_transform_two.T),
    )
    assert np.allclose(
        test.construct_array_lincomb(
            orb_transform_one, None, "cartesian", ["spherical", "cartesian"]
        ),
        orb_transform_one.dot(
            np.hstack(
                [
                    (np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2,
                    np.arange(9).reshape(3, 3) * 2,
                ]
            )
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(None, None, "cartesian", ["spherical", "cartesian"]),
        np.hstack(
            [(np.arange(9).reshape(3, 3)).dot(sph_transform.T) * 2, np.arange(9).reshape(3, 3) * 2]
        ),
    )


def test_construct_array_mix_missing_conventions():
    """Test BaseTwoIndexAsymmetric.construct_array_mix with partially defined conventions."""

    class SpecialShell(GeneralizedContractionShell):
        @property
        def angmom_components_sph(self):
            """Raise error in case undefined conventions are accessed."""
            raise NotImplementedError

    contractions = SpecialShell(1, np.array([1, 2, 3]), np.ones((1, 2)), np.ones(1))
    Test = disable_abstract(
        BaseTwoIndexAsymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont1, cont2, a=2: np.arange((2 * 3) ** 2, dtype=float).reshape(
                    2, 3, 2, 3
                )
                * a
            )
        },
    )
    test = Test([contractions, contractions], [contractions, contractions])
    assert np.allclose(
        test.construct_array_cartesian(a=3),
        test.construct_array_mix(["cartesian"] * 2, ["cartesian"] * 2, a=3),
    )
