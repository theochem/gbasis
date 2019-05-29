"""Test gbasis.moment."""
from gbasis._moment_int import _compute_multipole_moment_integrals
from gbasis.contractions import GeneralizedContractionShell, make_contractions
from gbasis.moment import Moment, moment_cartesian, moment_lincomb, moment_mix, moment_spherical
from gbasis.parsers import parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_moment_construct_array_contraction():
    """Test gbasis.moment.Moment.construct_array_contraction."""
    test_one = GeneralizedContractionShell(
        1, np.array([0.5, 1, 1.5]), np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    test_two = GeneralizedContractionShell(
        2, np.array([1.5, 2, 3]), np.array([3.0, 4.0]), np.array([0.2, 0.02])
    )
    answer = np.array(
        [
            [
                _compute_multipole_moment_integrals(
                    np.array([0, 0, 0]),
                    np.array(
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 2],
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                        ]
                    ),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components_cart
            ]
            for angmom_comp_one in test_one.angmom_components_cart
        ]
    )
    assert np.allclose(
        np.squeeze(
            Moment.construct_array_contraction(
                test_one,
                test_two,
                np.array([0, 0, 0]),
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [2, 0, 0],
                        [0, 2, 0],
                        [0, 0, 2],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                    ]
                ),
            )
        ),
        np.squeeze(answer),
    )

    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, None, np.array([0, 0, 0]), np.array([[0, 0, 0]])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            None, test_two, np.array([0, 0, 0]), np.array([[0, 0, 0]])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(test_one, test_two, [0, 1, 2], np.array([[0, 0, 0]]))
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, test_two, np.array([[0, 1, 2]]), np.array([[0, 0, 0]])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, test_two, np.array([0, 1, 2, 3]), np.array([[0, 0, 0]])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, test_two, np.array([0, 0, 0]), np.array([0, 0, 0])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, test_two, np.array([0, 0, 0]), np.array([[0, 0, 0, 0]])
        )
    with pytest.raises(TypeError):
        Moment.construct_array_contraction(
            test_one, test_two, np.array([0, 0, 0]), np.array([[0.0, 0.0, 0.0]])
        )


def test_moment_cartesian():
    """Test gbasis.moment.moment_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    moment_obj = Moment(basis)

    assert np.allclose(
        moment_obj.construct_array_cartesian(
            moment_coord=np.zeros(3),
            moment_orders=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
        moment_cartesian(
            basis,
            np.zeros(3),
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
    )


def test_moment_spherical():
    """Test gbasis.moment.moment_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    moment_obj = Moment(basis)
    assert np.allclose(
        moment_obj.construct_array_spherical(
            moment_coord=np.zeros(3),
            moment_orders=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
        moment_spherical(
            basis,
            np.zeros(3),
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
    )


def test_moment_mix():
    """Test gbasis.moment.moment_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    moment_obj = Moment(basis)
    assert np.allclose(
        moment_obj.construct_array_mix(
            ["spherical"] * 8,
            moment_coord=np.zeros(3),
            moment_orders=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
        moment_mix(
            basis,
            np.zeros(3),
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
            ["spherical"] * 8,
        ),
    )


def test_moment_spherical_lincomb():
    """Test gbasis.moment.moment_spherical_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    moment_obj = Moment(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        moment_obj.construct_array_lincomb(
            transform,
            "spherical",
            moment_coord=np.zeros(3),
            moment_orders=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
        moment_lincomb(
            basis,
            transform,
            np.zeros(3),
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
        ),
    )
