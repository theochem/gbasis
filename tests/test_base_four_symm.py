"""Test gbasis.base_four_symmetric."""
import numpy as np
import pytest
from utils import disable_abstract, skip_init

from gbasis.base_four_symm import BaseFourIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.spherical import generate_transformation


def test_init():
    """Test BaseFourIndexSymmetric.__init__."""
    Test = disable_abstract(BaseFourIndexSymmetric)
    test = skip_init(Test)
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    Test.__init__(test, [contractions])
    assert test._axes_contractions[0][0] == contractions
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions])


def test_contractions():
    """Test BaseFourIndexSymmetric.contractions."""
    Test = disable_abstract(BaseFourIndexSymmetric)
    cont = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical")
    test = Test([cont])
    assert test.contractions[0] == cont


def test_construct_array_contraction():
    """Test BaseFourIndexSymmetric.construct_array_contraction."""
    # enable only the abstract method construct_array_contraction
    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": BaseFourIndexSymmetric.construct_array_contraction
        },
    )
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    with pytest.raises(TypeError):
        Test([contractions])


def test_construct_array_cartesian():
    """Test BaseFourIndexSymmetric.construct_array_cartesian."""
    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones((1, 1)), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([2, 3, 4]), np.ones((1, 1)), 2 * np.ones(1), "spherical"
    )
    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont1, cont2, cont3, cont4: np.ones(
                (
                    cont1.num_seg_cont,
                    cont1.num_cart,
                    cont2.num_seg_cont,
                    cont2.num_cart,
                    cont3.num_seg_cont,
                    cont3.num_cart,
                    cont4.num_seg_cont,
                    cont4.num_cart,
                )
            )
            * cont1.exps
            * cont2.exps
            * cont3.exps
            * cont4.exps
        },
    )
    test = Test([cont_one, cont_two])
    answer = np.zeros(
        (cont_one.num_cart * cont_one.num_seg_cont + cont_two.num_cart * cont_two.num_seg_cont,) * 4
    )
    answer[:3, :3, :3, :3] = 1 * 1 * 1 * 1
    answer[:3, :3, :3, 3:9] = 1 * 1 * 1 * 2
    answer[:3, :3, 3:9, :3] = 1 * 1 * 2 * 1
    answer[:3, :3, 3:9, 3:9] = 1 * 1 * 2 * 2
    answer[:3, 3:9, :3, :3] = 1 * 2 * 1 * 1
    answer[:3, 3:9, :3, 3:9] = 1 * 2 * 1 * 2
    answer[:3, 3:9, 3:9, :3] = 1 * 2 * 2 * 1
    answer[:3, 3:9, 3:9, 3:9] = 1 * 2 * 2 * 2
    answer[3:9, :3, :3, :3] = 2 * 1 * 1 * 1
    answer[3:9, :3, :3, 3:9] = 2 * 1 * 1 * 2
    answer[3:9, :3, 3:9, :3] = 2 * 1 * 2 * 1
    answer[3:9, :3, 3:9, 3:9] = 2 * 1 * 2 * 2
    answer[3:9, 3:9, :3, :3] = 2 * 2 * 1 * 1
    answer[3:9, 3:9, :3, 3:9] = 2 * 2 * 1 * 2
    answer[3:9, 3:9, 3:9, :3] = 2 * 2 * 2 * 1
    answer[3:9, 3:9, 3:9, 3:9] = 2 * 2 * 2 * 2

    assert np.allclose(test.construct_array_cartesian(), answer)


def test_construct_array_spherical():
    """Test BaseFourIndexSymmetric.construct_array_spherical."""
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    transform = generate_transformation(
        1, contractions.angmom_components_cart, contractions.angmom_components_sph, "left"
    )

    # make symmetric
    array = np.arange(81, dtype=float).reshape(3, 3, 3, 3)
    array += np.einsum("ijkl->jikl", array)
    array += np.einsum("ijkl->ijlk", array)
    array += np.einsum("ijkl->klij", array)
    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, cont_three, cont_four, a=2: array.reshape(
                    1, 3, 1, 3, 1, 3, 1, 3
                )
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions])

    assert np.allclose(
        test.construct_array_spherical(),
        np.einsum("ijkl,ai,bj,ck,dl->abcd", array, transform, transform, transform, transform) * 2,
    )

    assert np.allclose(
        test.construct_array_spherical(a=3),
        np.einsum("ijkl,ai,bj,ck,dl->abcd", array, transform, transform, transform, transform) * 3,
    )
    with pytest.raises(TypeError):
        test.construct_array_spherical(bad_keyword=3)

    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    transform_one = generate_transformation(
        1, cont_one.angmom_components_cart, cont_one.angmom_components_sph, "left"
    )
    transform_two = generate_transformation(
        2, cont_two.angmom_components_cart, cont_two.angmom_components_sph, "left"
    )

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, cont_three, cont_four: (
                np.arange(
                    cont_one.num_cart
                    * cont_two.num_cart
                    * cont_three.num_cart
                    * cont_four.num_cart
                    * 2,
                    dtype=float,
                ).reshape(
                    1,
                    cont_one.num_cart,
                    1,
                    cont_two.num_cart,
                    1,
                    cont_three.num_cart,
                    1,
                    cont_four.num_cart,
                    2,
                )
            )
        },
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))
    test = Test([cont_one, cont_two])
    # NOTE: since the test subarray (output of construct_array_contraction) does not satisfy the
    # symmetries of the two electron integral, only the last permutation is used. If this output
    # satisfies the symmetries of two electron integrals, then all these permutations should result
    # in the same array.
    # FIXME: not a good test
    assert np.allclose(
        test.construct_array_spherical()[:3, :3, :3, :3],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 3 * 3 * 2).reshape(3, 3, 3, 3, 2),
                transform_one,
                transform_one,
                transform_one,
                transform_one,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, :3, :3, 3:],
        np.einsum(
            "ijklm->jiklm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 3 * 6 * 2).reshape(3, 3, 3, 6, 2),
                transform_one,
                transform_one,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, :3, 3:, :3],
        np.einsum(
            "ijklm->jilkm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 3 * 6 * 2).reshape(3, 3, 3, 6, 2),
                transform_one,
                transform_one,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, :3, 3:, 3:],
        np.einsum(
            "ijklm->jilkm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 6 * 6 * 2).reshape(3, 3, 6, 6, 2),
                transform_one,
                transform_one,
                transform_two,
                transform_two,
            ),
        ),
    )

    assert np.allclose(
        test.construct_array_spherical()[:3, 3:, :3, :3],
        np.einsum(
            "ijklm->kljim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 3 * 6 * 2).reshape(3, 3, 3, 6, 2),
                transform_one,
                transform_one,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, 3:, :3, 3:],
        np.einsum(
            "ijklm->klijm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 3 * 6 * 2).reshape(3, 6, 3, 6, 2),
                transform_one,
                transform_two,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, 3:, 3:, :3],
        np.einsum(
            "ijklm->kljim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 3 * 6 * 2).reshape(3, 6, 3, 6, 2),
                transform_one,
                transform_two,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[:3, 3:, 3:, 3:],
        np.einsum(
            "ijklm->ijlkm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 6 * 6 * 2).reshape(3, 6, 6, 6, 2),
                transform_one,
                transform_two,
                transform_two,
                transform_two,
            ),
        ),
    )

    assert np.allclose(
        test.construct_array_spherical()[3:, :3, :3, :3],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 3 * 6 * 2).reshape(3, 3, 3, 6, 2),
                transform_one,
                transform_one,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, :3, :3, 3:],
        np.einsum(
            "ijklm->lkijm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 3 * 6 * 2).reshape(3, 6, 3, 6, 2),
                transform_one,
                transform_two,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, :3, 3:, :3],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 3 * 6 * 2).reshape(3, 6, 3, 6, 2),
                transform_one,
                transform_two,
                transform_one,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, :3, 3:, 3:],
        np.einsum(
            "ijklm->jilkm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 6 * 6 * 2).reshape(3, 6, 6, 6, 2),
                transform_one,
                transform_two,
                transform_two,
                transform_two,
            ),
        ),
    )

    assert np.allclose(
        test.construct_array_spherical()[3:, 3:, :3, :3],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 3 * 6 * 6 * 2).reshape(3, 3, 6, 6, 2),
                transform_one,
                transform_one,
                transform_two,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, 3:, :3, 3:],
        np.einsum(
            "ijklm->lkijm",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 6 * 6 * 2).reshape(3, 6, 6, 6, 2),
                transform_one,
                transform_two,
                transform_two,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, 3:, 3:, :3],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(3 * 6 * 6 * 6 * 2).reshape(3, 6, 6, 6, 2),
                transform_one,
                transform_two,
                transform_two,
                transform_two,
            ),
        ),
    )
    assert np.allclose(
        test.construct_array_spherical()[3:, 3:, 3:, 3:],
        np.einsum(
            "ijklm->lkjim",
            np.einsum(
                "ijklm,ai,bj,ck,dl->abcdm",
                np.arange(6 * 6 * 6 * 6 * 2).reshape(6, 6, 6, 6, 2),
                transform_two,
                transform_two,
                transform_two,
                transform_two,
            ),
        ),
    )


def test_construct_array_mix():
    """Test BaseFourIndex.construct_array_mix."""
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, cont_three, cont_four, a=2: np.arange(
                    81, dtype=float
                ).reshape(1, 3, 1, 3, 1, 3, 1, 3)
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

    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, cont_three, cont_four: (
                np.arange(
                    cont_one.num_cart
                    * cont_two.num_cart
                    * cont_three.num_cart
                    * cont_four.num_cart
                    * 2,
                    dtype=float,
                ).reshape(
                    1,
                    cont_one.num_cart,
                    1,
                    cont_two.num_cart,
                    1,
                    cont_three.num_cart,
                    1,
                    cont_four.num_cart,
                    2,
                )
            )
        },
    )
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2)
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2)
    )

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, cont_three, cont_four: (
                np.arange(
                    2
                    * cont_one.num_cart
                    * 2
                    * cont_two.num_cart
                    * 2
                    * cont_three.num_cart
                    * 2
                    * cont_four.num_cart
                    * 2,
                    dtype=float,
                ).reshape(
                    2,
                    cont_one.num_cart,
                    2,
                    cont_two.num_cart,
                    2,
                    cont_three.num_cart,
                    2,
                    cont_four.num_cart,
                    2,
                )
            )
        },
    )
    cont_one.norm_cont = np.ones((2, cont_one.num_cart))
    cont_two.norm_cont = np.ones((2, cont_two.num_cart))
    test = Test([cont_one, cont_two])
    assert np.allclose(
        test.construct_array_spherical(), test.construct_array_mix(["spherical"] * 2)
    )
    assert np.allclose(
        test.construct_array_cartesian(), test.construct_array_mix(["cartesian"] * 2)
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


def test_construct_array_mix_with_both_cartesian_and_spherical():
    r"""Test construct_array_mix with both a P-Type Cartesian and D-Type Spherical contractions."""
    num_pts, num_seg_shell = 1, 1
    # Define the coefficients used to seperate which contraction block it is,
    #       Put it in a dictionary to avoid doing so many nested if-statements.
    coeff_p_p_p_p_type = 2
    coeff_p_p_p_d_type = 4
    coeff_p_p_d_d_type = 5
    coeff_p_d_p_p_type = 6
    coeff_p_d_p_d_type = 8
    coeff_p_d_d_d_type = 10
    coeff_d_d_p_p_type = 12
    coeff_d_d_p_d_type = 14
    coeff_d_d_d_d_type = 16
    coeff_dict = {
        "1111": coeff_p_p_p_p_type,
        "1112": coeff_p_p_p_d_type,
        "1122": coeff_p_p_d_d_type,
        "1211": coeff_p_d_p_p_type,
        "1212": coeff_p_d_p_d_type,
        "1222": coeff_p_d_d_d_type,
        "2211": coeff_d_d_p_p_type,
        "2212": coeff_d_d_p_d_type,
        "2222": coeff_d_d_d_d_type,
    }

    def construct_array_cont(self, cont_one, cont_two, cont_three, cont_four):
        output = np.ones(
            cont_one.num_cart
            * cont_two.num_cart
            * cont_three.num_cart
            * cont_four.num_cart
            * num_pts,
            dtype=float,
        ).reshape(
            num_seg_shell,
            cont_one.num_cart,
            num_seg_shell,
            cont_two.num_cart,
            num_seg_shell,
            cont_three.num_cart,
            num_seg_shell,
            cont_four.num_cart,
            num_pts,
        )
        identifier = (
            str(cont_one.angmom)
            + str(cont_two.angmom)
            + str(cont_three.angmom)
            + str(cont_four.angmom)
        )
        return output * coeff_dict[identifier]

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={"construct_array_contraction": construct_array_cont},
    )
    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )

    # Remove the dependence on norm constants.
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))
    test = Test([cont_one, cont_two])

    # Should have shape (3 + 5, 3 + 5, NUM_PTS), due to the following:
    #           3-> Number of P-type, 5->Number of Spherical D-type.
    actual = test.construct_array_mix(["cartesian", "spherical"])[:, :, :, :, 0]

    # Test P-type to P-type to P-Type To P-type i.e. (P, P, P, P)
    assert np.allclose(actual[:3, :3, :3, :3], np.ones((3, 3)) * coeff_p_p_p_p_type)
    # Test (P, P, P, D)
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
        actual[:3, :3, :3, 3:],
        np.ones((3, 3, 3, 6)).dot(generate_transformation_array.T) * coeff_p_p_p_d_type,
    )

    assert np.allclose(
        actual[:3, :3, 3:, :3],
        np.einsum("ij,mnjl->mnil", generate_transformation_array, np.ones((3, 3, 6, 3)))
        * coeff_p_p_p_d_type,
    )
    # Test (P, P, D, D), (D, D, P, P)
    assert np.allclose(
        actual[:3, :3, 3:, 3:],
        np.einsum(
            "ij,mnjl,pl->mnip",
            generate_transformation_array,
            np.ones((3, 3, 6, 6)),
            generate_transformation_array,
        )
        * coeff_p_p_d_d_type,
    )
    assert np.allclose(actual[3:, 3:, :3, :3], actual[:3, :3, 3:, 3:].T)  # Symmetry
    # Test (P, D, P, D)
    assert np.allclose(
        actual[:3, 3:, :3, 3:],
        np.einsum(
            "ij,mjnl,pl->minp",
            generate_transformation_array,
            np.ones((3, 6, 3, 6)),
            generate_transformation_array,
        )
        * coeff_p_d_p_d_type,
    )
    # Test (P, D, D, D), & (D, D, P, D)
    assert np.allclose(
        actual[:3, 3:, 3:, 3:],
        np.einsum(
            "in,mnjl,pl,oj->miop",
            generate_transformation_array,
            np.ones((3, 6, 6, 6)),
            generate_transformation_array,
            generate_transformation_array,
        )
        * coeff_p_d_d_d_type,
    )
    assert np.allclose(actual[3:, 3:, :3, 3:], np.einsum("ijkl->klij", actual[:3, 3:, 3:, 3:]))
    # Test (D, D, D, D)
    assert np.allclose(
        actual[3:, 3:, 3:, 3:],
        np.einsum(
            "dm,in,mnjl,pl,oj->diop",
            generate_transformation_array,
            generate_transformation_array,
            np.ones((6, 6, 6, 6)),
            generate_transformation_array,
            generate_transformation_array,
        )
        * coeff_d_d_d_d_type,
    )


def test_construct_array_lincomb():
    """Test BaseFourIndexSymmetric.construct_array_lincomb."""
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    sph_transform = generate_transformation(
        1, contractions.angmom_components_cart, contractions.angmom_components_sph, "left"
    )
    orb_transform = np.random.rand(3, 3)

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont_one, cont_two, cont_three, cont_four, a=2: np.arange(
                    81, dtype=float
                ).reshape(1, 3, 1, 3, 1, 3, 1, 3)
                * a
            )
        },
    )
    contractions.norm_cont = np.ones((1, 3))
    test = Test([contractions])
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, ["cartesian"]),
        np.einsum(
            "ijkl,ai,bj,ck,dl->abcd",
            np.einsum("ijkl->lkji", np.arange(81).reshape(3, 3, 3, 3)) * 2,
            orb_transform,
            orb_transform,
            orb_transform,
            orb_transform,
        ),
    )
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, ["spherical"]),
        np.einsum(
            "ijkl,ai,bj,ck,dl->abcd",
            np.einsum(
                "ijkl,ai,bj,ck,dl->abcd",
                np.einsum("ijkl->lkji", np.arange(81).reshape(3, 3, 3, 3)) * 2,
                sph_transform,
                sph_transform,
                sph_transform,
                sph_transform,
            ),
            orb_transform,
            orb_transform,
            orb_transform,
            orb_transform,
        ),
    )
    with pytest.raises(TypeError):
        test.construct_array_lincomb(orb_transform, "spherical", bad_keyword=3)
    with pytest.raises(TypeError):
        test.construct_array_lincomb(orb_transform, "bad", keyword=3)

    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": lambda self, cont_one, cont_two, cont_three, cont_four: (
                np.arange(
                    cont_one.num_cart
                    * cont_two.num_cart
                    * cont_three.num_cart
                    * cont_four.num_cart,
                    dtype=float,
                ).reshape(
                    1,
                    cont_one.num_cart,
                    1,
                    cont_two.num_cart,
                    1,
                    cont_three.num_cart,
                    1,
                    cont_four.num_cart,
                )
            )
        },
    )
    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))
    test = Test([cont_one, cont_two])

    sph_transform_one = generate_transformation(
        1, cont_one.angmom_components_cart, cont_one.angmom_components_sph, "left"
    )
    sph_transform_two = generate_transformation(
        2, cont_two.angmom_components_cart, cont_two.angmom_components_sph, "left"
    )
    orb_transform = np.random.rand(8, 8)
    # NOTE: since the test subarray (output of construct_array_contraction) does not satisfy the
    # symmetries of the two electron integral, only the last permutation is used. If this output
    # satisfies the symmetries of two electron integrals, then all these permutations should result
    # in the same array.
    # FIXME: not a good test
    sph_array = np.concatenate(
        [
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jikl",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 6 * 6).reshape(3, 3, 6, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->klji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->klij",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->klji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->ijlk",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkij",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_one,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 6 * 6).reshape(3, 3, 6, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkij",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(6 * 6 * 6 * 6).reshape(6, 6, 6, 6),
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                            sph_transform_two,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                ],
                axis=1,
            ),
        ]
    )

    assert np.allclose(
        test.construct_array_lincomb(orb_transform, ["spherical"]),
        np.einsum(
            "ijkl,ai,bj,ck,dl->abcd",
            sph_array,
            orb_transform,
            orb_transform,
            orb_transform,
            orb_transform,
        ),
    )

    orb_transform = np.random.rand(9, 9)
    sph_cart_array = np.concatenate(
        [
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck,dl->abcd",
                                            np.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jikl",
                                        np.einsum(
                                            "ijkl,ai,bj,ck->abcl",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai,bj,ck->abcl",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai,bj->abkl",
                                            np.arange(3 * 3 * 6 * 6).reshape(3, 3, 6, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->klji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck->abcl",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->klij",
                                        np.einsum(
                                            "ijkl,ai,ck->ajcl",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->klji",
                                        np.einsum(
                                            "ijkl,ai,ck->ajcl",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->ijlk",
                                        np.einsum(
                                            "ijkl,ai->ajkl",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj,ck->abcl",
                                            np.arange(3 * 3 * 3 * 6).reshape(3, 3, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkij",
                                        np.einsum(
                                            "ijkl,ai,ck->ajcl",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,ck->ajcl",
                                            np.arange(3 * 6 * 3 * 6).reshape(3, 6, 3, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->jilk",
                                        np.einsum(
                                            "ijkl,ai->ajkl",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                    np.concatenate(
                        [
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai,bj->abkl",
                                            np.arange(3 * 3 * 6 * 6).reshape(3, 3, 6, 6),
                                            sph_transform_one,
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkij",
                                        np.einsum(
                                            "ijkl,ai->ajkl",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                        ),
                                    ),
                                ],
                                axis=3,
                            ),
                            np.concatenate(
                                [
                                    np.einsum(
                                        "ijkl->lkji",
                                        np.einsum(
                                            "ijkl,ai->ajkl",
                                            np.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6),
                                            sph_transform_one,
                                        ),
                                    ),
                                    np.einsum(
                                        "ijkl->lkji", np.arange(6 * 6 * 6 * 6).reshape(6, 6, 6, 6)
                                    ),
                                ],
                                axis=3,
                            ),
                        ],
                        axis=2,
                    ),
                ],
                axis=1,
            ),
        ]
    )
    assert np.allclose(
        test.construct_array_lincomb(orb_transform, ("spherical", "cartesian")),
        np.einsum(
            "ijkl,ai,bj,ck,dl->abcd",
            sph_cart_array,
            orb_transform,
            orb_transform,
            orb_transform,
            orb_transform,
        ),
    )


def test_construct_array_mix_missing_conventions():
    """Test BaseFourIndexSymmetric.construct_array_mix with partially defined conventions."""

    class SpecialShell(GeneralizedContractionShell):
        @property
        def angmom_components_sph(self):
            """Raise error in case undefined conventions are accessed."""
            raise NotImplementedError

    contractions = SpecialShell(1, np.array([1, 2, 3]), np.ones((1, 2)), np.ones(1), "spherical")
    Test = disable_abstract(
        BaseFourIndexSymmetric,
        dict_overwrite={
            "construct_array_contraction": (
                lambda self, cont1, cont2, cont3, cont4, a=2: np.arange(
                    (2 * 3) ** 4, dtype=float
                ).reshape(2, 3, 2, 3, 2, 3, 2, 3)
                * a
            )
        },
    )
    test = Test([contractions, contractions])
    assert np.allclose(
        test.construct_array_cartesian(a=3), test.construct_array_mix(["cartesian"] * 2, a=3)
    )
