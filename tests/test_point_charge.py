"""Test gbasis.point_charge."""
from gbasis.contractions import ContractedCartesianGaussians, make_contractions
from gbasis.parsers import parse_nwchem
from gbasis.point_charge import (
    point_charge_cartesian,
    point_charge_lincomb,
    point_charge_mix,
    point_charge_spherical,
    PointChargeIntegral,
)
import numpy as np
import pytest
from scipy.special import factorial
from utils import find_datafile


def boys_helgaker(n, x):
    """Return the Boys function as written in Helgaker, eq. 9.8.39.

    References
    ----------
    See Helgaker, T., Jorgensen, P., Olsen, J. Molecular Electronic Structure Theory. Wiley (2013),
    pg 372.

    """
    return sum((-x) ** k / factorial(k) / (2 * n + 2 * k + 1) for k in range(100))


def test_boys_func():
    """Test gbasis.point_charge.PointChargeIntegral.boys_func."""
    orders = np.arange(10)
    distances = np.random.rand(20, 30)
    test = PointChargeIntegral.boys_func(orders[:, None, None], distances[None, :, :])

    ref = np.array([boys_helgaker(i, j) for i in orders for j in distances.ravel()])
    assert np.allclose(test, ref.reshape(10, 20, 30))


def test_construct_array_contraction():
    """Test gbasis.point_charge.PointChargeIntegral.construct_array_contraction."""
    coord_one = np.array([0.5, 1, 1.5])
    test_one = ContractedCartesianGaussians(0, coord_one, 0, np.array([1.0]), np.array([0.1]))
    coord_two = np.array([1.5, 2, 3])
    test_two = ContractedCartesianGaussians(0, coord_two, 0, np.array([3.0]), np.array([0.2]))
    coord = np.array([[0, 0, 0]])
    charge = np.array([1])
    coord_wac = (0.1 * coord_one + 0.2 * coord_two) / (0.1 + 0.2)

    with pytest.raises(TypeError):
        PointChargeIntegral.construct_array_contraction(None, test_two, coord, charge)
    with pytest.raises(TypeError):
        PointChargeIntegral.construct_array_contraction(test_one, None, coord, charge)
    with pytest.raises(TypeError):
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord.tolist(), charge)
    with pytest.raises(TypeError):
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord.ravel(), charge)
    with pytest.raises(TypeError):
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord, float(charge))
    with pytest.raises(ValueError):
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord, np.array([0, 1]))

    assert np.allclose(
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord, charge),
        (
            2
            * np.pi
            / (0.1 + 0.2)
            * boys_helgaker(0, (0.1 + 0.2) * np.sum(coord_wac ** 2))
            * np.exp(-0.1 * 0.2 / (0.1 + 0.2) * np.sum((coord_one - coord_two) ** 2))
            * (2 * 0.1 / np.pi) ** (3 / 4)
            * (2 * 0.2 / np.pi) ** (3 / 4)
            * 1
            * 3
            * (-1)
        ),
    )

    test_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    test_two = ContractedCartesianGaussians(
        0, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.2])
    )
    v_000_000 = [
        2
        * np.pi
        / (0.1 + 0.2)
        * boys_helgaker(i, (0.1 + 0.2) * np.sum(coord_wac ** 2))
        * np.exp(-0.1 * 0.2 / (0.1 + 0.2) * np.sum((coord_one - coord_two) ** 2))
        for i in range(2)
    ]
    assert np.allclose(
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord, charge).ravel(),
        (
            ((coord_wac - coord_one) * v_000_000[0] - (coord_wac - coord[0]) * v_000_000[1])
            * (2 * 0.1 / np.pi) ** (3 / 4)
            * ((4 * 0.1) ** 0.5)
            * (2 * 0.2 / np.pi) ** (3 / 4)
            * 1
            * 3
            * (-1)
        )[::-1],
    )

    test_one = ContractedCartesianGaussians(
        0, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    test_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.2])
    )
    assert np.allclose(
        PointChargeIntegral.construct_array_contraction(test_one, test_two, coord, charge).ravel(),
        (
            ((coord_wac - coord_two) * v_000_000[0] - (coord_wac - coord[0]) * v_000_000[1])
            * (2 * 0.1 / np.pi) ** (3 / 4)
            * (2 * 0.2 / np.pi) ** (3 / 4)
            * ((4 * 0.2) ** 0.5)
            * 1
            * 3
            * (-1)
        )[::-1],
    )


def test_point_charge_cartesian():
    """Test gbasis.point_charge.point_charge_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    point_charge_obj = PointChargeIntegral(basis)

    coords_points = np.random.rand(5, 3)
    charges_points = np.random.rand(5)
    assert np.allclose(
        point_charge_obj.construct_array_cartesian(
            coords_points=coords_points, charges_points=charges_points
        ),
        point_charge_cartesian(basis, coords_points=coords_points, charges_points=charges_points),
    )


def test_point_charge_spherical():
    """Test gbasis.point_charge.point_charge_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    point_charge_obj = PointChargeIntegral(basis)

    coords_points = np.random.rand(5, 3)
    charges_points = np.random.rand(5)
    assert np.allclose(
        point_charge_obj.construct_array_spherical(
            coords_points=coords_points, charges_points=charges_points
        ),
        point_charge_spherical(basis, coords_points=coords_points, charges_points=charges_points),
    )


def test_point_charge_mix():
    """Test gbasis.point_charge.point_charge_mix."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    point_charge_obj = PointChargeIntegral(basis)

    coords_points = np.random.rand(5, 3)
    charges_points = np.random.rand(5)
    assert np.allclose(
        point_charge_obj.construct_array_mix(
            ["spherical"] * 8, coords_points=coords_points, charges_points=charges_points
        ),
        point_charge_mix(basis, coords_points, charges_points, ["spherical"] * 8),
    )


def test_point_charge_lincomb():
    """Test gbasis.point_charge.point_charge_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    point_charge_obj = PointChargeIntegral(basis)

    coords_points = np.random.rand(5, 3)
    charges_points = np.random.rand(5)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        point_charge_obj.construct_array_lincomb(
            transform, ["spherical"] * 8, coords_points=coords_points, charges_points=charges_points
        ),
        point_charge_lincomb(
            basis, transform, coords_points=coords_points, charges_points=charges_points
        ),
    )
