"""Test gbasis.spherical."""
import itertools as it

from gbasis.spherical import (
    expansion_coeff,
    generate_transformation,
    harmonic_norm,
    real_solid_harmonic,
    shift_factor,
)
import numpy as np
import pytest


def test_shift_factor():
    """Test spherical.shift_factor."""
    assert shift_factor(2) == 0.0
    assert shift_factor(0) == 0.0
    assert shift_factor(-1) == 0.5
    with pytest.raises(TypeError):
        shift_factor(0.5)
    with pytest.raises(TypeError):
        shift_factor(None)


def test_expansion_coeff():
    """Test spherical.expansion_coeff."""
    assert expansion_coeff(0, 0, 0, 0, 0) == 1.0
    assert expansion_coeff(1, 0, 0, 0, 0) == 1.0
    with pytest.raises(TypeError):
        expansion_coeff(0.0, 0, 0, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0.0, 0, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0, 0.0, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0, 0, 0.0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(None, 0, 0, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, None, 0, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0, None, 0, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0, 0, None, 0)
    with pytest.raises(TypeError):
        expansion_coeff(0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        expansion_coeff(-1, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        expansion_coeff(3, 4, 0, 0, 0)
    with pytest.raises(ValueError):
        expansion_coeff(1, -2, 0, 0, 0)
    with pytest.raises(ValueError):
        expansion_coeff(2, -1, 0, 0, 0)
    with pytest.raises(ValueError):
        expansion_coeff(2, 1, 0, 0, 0.5)


def test_harmonic_norm():
    """Test spherical.harmonic_norm."""
    assert harmonic_norm(0, 0) == 1
    with pytest.raises(TypeError):
        harmonic_norm(0.0, 0)
    with pytest.raises(TypeError):
        harmonic_norm(0, 0.0)
    with pytest.raises(TypeError):
        harmonic_norm(None, 0)
    with pytest.raises(TypeError):
        harmonic_norm(0, None)
    with pytest.raises(ValueError):
        harmonic_norm(-1, 0)
    with pytest.raises(ValueError):
        harmonic_norm(1, 2)
    with pytest.raises(ValueError):
        harmonic_norm(0, -1)


def test_real_solid_harmonic():
    """Test spherical.real_solid_harmonic.

    All real solid harmonics obtained from Helgaker et al. "Molecular Electronic-Structure Theory",
    pg. 211 (Table 6.3).
    """
    assert real_solid_harmonic(0, 0) == {(0, 0, 0): 1.0}
    assert real_solid_harmonic(1, 1) == {(1, 0, 0): 1.0}
    assert real_solid_harmonic(1, 0) == {(0, 0, 1): 1.0}
    assert real_solid_harmonic(1, -1) == {(0, 1, 0): 1.0}
    assert real_solid_harmonic(2, 2) == {(0, 2, 0): -np.sqrt(3) / 2, (2, 0, 0): np.sqrt(3) / 2}
    assert real_solid_harmonic(2, 1) == {(1, 0, 1): np.sqrt(3)}
    assert real_solid_harmonic(2, 0) == {(0, 0, 2): 1.0, (2, 0, 0): -0.5, (0, 2, 0): -0.5}
    assert real_solid_harmonic(2, -1) == {(0, 1, 1): np.sqrt(3)}
    assert real_solid_harmonic(2, -2) == {(1, 1, 0): np.sqrt(3)}
    with pytest.raises(TypeError):
        real_solid_harmonic(0.0, 0)
    with pytest.raises(TypeError):
        real_solid_harmonic(0, 0.0)
    with pytest.raises(TypeError):
        real_solid_harmonic(None, 0)
    with pytest.raises(TypeError):
        real_solid_harmonic(0, None)
    with pytest.raises(ValueError):
        real_solid_harmonic(-1, 0)
    with pytest.raises(ValueError):
        real_solid_harmonic(1, 2)
    with pytest.raises(ValueError):
        real_solid_harmonic(0, -1)


def test_generate_transformation():
    """Test spherical.generate_transformation."""
    assert np.array_equal(
        generate_transformation(0, np.array([(0, 0, 0)]), (0,), "right"), np.array([[1.0]])
    )
    assert np.array_equal(
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "right"
        ),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert np.array_equal(
        generate_transformation(0, np.array([(0, 0, 0)]), (0,), "right").T,
        generate_transformation(0, np.array([(0, 0, 0)]), (0,), "left"),
    )
    assert np.array_equal(
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "right"
        ).T,
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "left"),
    )
    with pytest.raises(TypeError):
        generate_transformation(0.0, np.array([(0, 0, 0)]), (0,), "right")
    with pytest.raises(TypeError):
        generate_transformation(0, 0, (0,), "right")
    with pytest.raises(ValueError):
        generate_transformation(
            -1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "right"
        )
    with pytest.raises(ValueError):
        generate_transformation(0, np.array([(0, 0, 0, 0)]), (0,), "right")
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 1)]), (-1, 0, 1), "right")
    with pytest.raises(ValueError):
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]), (-1, 0, 1), "right"
        )
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0)]), (-1, 0, 1), "right")
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 0, 0), (0, 0, 2)]), (-1, 0, 1), "right")
    with pytest.raises(TypeError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), 1)
    with pytest.raises(TypeError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), None)
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "up")
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1), "")
    # check angmom_components_sph type
    with pytest.raises(TypeError):
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), np.array([-1, 0, 1]), "left"
        )
    with pytest.raises(TypeError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), {-1, 0, 1}, "left")
    with pytest.raises(TypeError):
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), [-1, 0, 1.0], "left"
        )
    # check angmom_components_sph value
    with pytest.raises(ValueError):
        generate_transformation(
            1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 0, 1, 1), "left"
        )
    with pytest.raises(ValueError):
        generate_transformation(1, np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), (-1, 2, 1), "left")


# FIXME: cannot reproduce horton results for the angular momentum 4. I  have a feeling that the
# horton results for the are incorrect here, but it will need to be double checked.
def test_generate_transformation_horton():
    """Test spherical.generate_transformation using horton reference.

    Answer obtained from https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html
    and https://github.com/theochem/horton/blob/master/horton/gbasis/cartpure.cpp.

    """
    answer = np.array([[1]])
    assert np.allclose(generate_transformation(0, np.array([[0, 0, 0]]), (0,), "left"), answer)

    answer = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(
        generate_transformation(1, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (0, 1, -1), "left"),
        answer,
    )

    answer = np.array(
        [
            [-0.5, 0, 0, -0.5, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0.5 * 3 ** 0.5, 0, 0, -0.5 * 3 ** 0.5, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(
        generate_transformation(
            2,
            np.array([[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]),
            (0, 1, -1, 2, -2),
            "left",
        ),
        answer,
    )

    answer = np.array(
        [
            [0, 0, -3 / 10 * 5 ** 0.5, 0, 0, 0, 0, -3 / 10 * 5 ** 0.5, 0, 1],
            [-1 / 4 * 6 ** 0.5, 0, 0, -1 / 20 * 30 ** 0.5, 0, 1 / 5 * 30 ** 0.5, 0, 0, 0, 0],
            [0, -1 / 20 * 30 ** 0.5, 0, 0, 0, 0, -1 / 4 * 6 ** 0.5, 0, 1 / 5 * 30 ** 0.5, 0],
            [0, 0, 1 / 2 * 3 ** 0.5, 0, 0, 0, 0, -1 / 2 * 3 ** 0.5, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1 / 4 * 10 ** 0.5, 0, 0, -3 / 4 * 2 ** 0.5, 0, 0, 0, 0, 0, 0],
            [0, 3 / 4 * 2 ** 0.5, 0, 0, 0, 0, -1 / 4 * 10 ** 0.5, 0, 0, 0],
        ]
    )
    # shuffle to have correct order
    assert np.allclose(
        generate_transformation(
            3,
            np.array(
                [
                    [3, 0, 0],
                    [2, 1, 0],
                    [2, 0, 1],
                    [1, 2, 0],
                    [1, 1, 1],
                    [1, 0, 2],
                    [0, 3, 0],
                    [0, 2, 1],
                    [0, 1, 2],
                    [0, 0, 3],
                ]
            ),
            (0, 1, -1, 2, -2, 3, -3),
            "left",
        ),
        answer,
    )

    # taken from HORTON's gbasis/cartpure.cpp
    # horton_transform = [
    #     [0, 0, 0.375],
    #     [0, 3, 0.21957751641341996535],
    #     [0, 5, -0.87831006565367986142],
    #     [0, 10, 0.375],
    #     [0, 12, -0.87831006565367986142],
    #     [0, 14, 1.0],
    #     [1, 2, -0.89642145700079522998],
    #     [1, 7, -0.40089186286863657703],
    #     [1, 9, 1.19522860933439364],
    #     [2, 4, -0.40089186286863657703],
    #     [2, 11, -0.89642145700079522998],
    #     [2, 13, 1.19522860933439364],
    #     [3, 0, -0.5590169943749474241],
    #     [3, 5, 0.9819805060619657157],
    #     [3, 10, 0.5590169943749474241],
    #     [3, 12, -0.9819805060619657157],
    #     [4, 1, -0.42257712736425828875],
    #     [4, 6, -0.42257712736425828875],
    #     [4, 8, 1.1338934190276816816],
    #     [5, 2, 0.790569415042094833],
    #     [5, 7, -1.0606601717798212866],
    #     [6, 4, 1.0606601717798212866],
    #     [6, 11, -0.790569415042094833],
    #     [7, 0, 0.73950997288745200532],
    #     [7, 3, -1.2990381056766579701],
    #     [7, 10, 0.73950997288745200532],
    #     [8, 1, 1.1180339887498948482],
    #     [8, 6, -1.1180339887498948482],
    # ]
    # answer = np.zeros((9, 15))
    # for i in horton_transform:
    #     answer[i[0], i[1]] = i[2]

    # assert np.allclose(
    #     generate_transformation(
    #         4,
    #         np.array(
    #             [
    #                 [4, 0, 0],
    #                 [3, 1, 0],
    #                 [3, 0, 1],
    #                 [2, 2, 0],
    #                 [2, 1, 1],
    #                 [2, 0, 2],
    #                 [1, 3, 0],
    #                 [1, 2, 1],
    #                 [1, 1, 2],
    #                 [1, 0, 3],
    #                 [0, 4, 0],
    #                 [0, 3, 1],
    #                 [0, 2, 2],
    #                 [0, 1, 3],
    #                 [0, 0, 4],
    #             ]
    #         ),
    #         (0, 1, -1, 2, -2, 3, -3, 4, -4),
    #         "left",
    #     ),
    #     answer,
    # )

    # taken from HORTON's gbasis/cartpure.cpp
    # horton_transform = [
    #     [0, 2, 0.625],
    #     [0, 7, 0.36596252735569994226],
    #     [0, 9, -1.0910894511799619063],
    #     [0, 16, 0.625],
    #     [0, 18, -1.0910894511799619063],
    #     [0, 20, 1.0],
    #     [1, 0, 0.48412291827592711065],
    #     [1, 3, 0.21128856368212914438],
    #     [1, 5, -1.2677313820927748663],
    #     [1, 10, 0.16137430609197570355],
    #     [1, 12, -0.56694670951384084082],
    #     [1, 14, 1.2909944487358056284],
    #     [2, 1, 0.16137430609197570355],
    #     [2, 6, 0.21128856368212914438],
    #     [2, 8, -0.56694670951384084082],
    #     [2, 15, 0.48412291827592711065],
    #     [2, 17, -1.2677313820927748663],
    #     [2, 19, 1.2909944487358056284],
    #     [3, 2, -0.85391256382996653193],
    #     [3, 9, 1.1180339887498948482],
    #     [3, 16, 0.85391256382996653193],
    #     [3, 18, -1.1180339887498948482],
    #     [4, 4, -0.6454972243679028142],
    #     [4, 11, -0.6454972243679028142],
    #     [4, 13, 1.2909944487358056284],
    #     [5, 0, -0.52291251658379721749],
    #     [5, 3, 0.22821773229381921394],
    #     [5, 5, 0.91287092917527685576],
    #     [5, 10, 0.52291251658379721749],
    #     [5, 12, -1.2247448713915890491],
    #     [6, 1, -0.52291251658379721749],
    #     [6, 6, -0.22821773229381921394],
    #     [6, 8, 1.2247448713915890491],
    #     [6, 15, 0.52291251658379721749],
    #     [6, 17, -0.91287092917527685576],
    #     [7, 2, 0.73950997288745200532],
    #     [7, 7, -1.2990381056766579701],
    #     [7, 16, 0.73950997288745200532],
    #     [8, 4, 1.1180339887498948482],
    #     [8, 11, -1.1180339887498948482],
    #     [9, 0, 0.7015607600201140098],
    #     [9, 3, -1.5309310892394863114],
    #     [9, 10, 1.169267933366856683],
    #     [10, 1, 1.169267933366856683],
    #     [10, 6, -1.5309310892394863114],
    #     [10, 15, 0.7015607600201140098],
    # ]
    # answer = np.zeros((11, 21))
    # for i in horton_transform:
    #     answer[i[0], i[1]] = i[2]

    # assert np.allclose(
    #     generate_transformation(
    #         5,
    #         np.array(
    #             [
    #                 (i.count(0), i.count(1), i.count(2))
    #                 for i in it.combinations_with_replacement(range(3), 5)
    #             ]
    #         ),
    #         (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5),
    #         "left",
    #     ),
    #     answer,
    # )
