"""Test gbasis.contractions."""
from gbasis.contractions import cartesian_gaussian_norm, ContractedCartesianGaussians
import numpy as np
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


def test_coord_setter():
    """Test setter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test.coord = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )
    test.coord = np.array([1, 2, 3])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )

    with pytest.raises(TypeError):
        test.coord = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2])
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2, 3], dtype=bool)


def test_coord_getter():
    """Test getter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test._coord = 2
    assert test.coord == 2


def test_angmom_setter():
    """Test setter for ContractedCartesianGaussians.angmom."""
    test = skip_init(ContractedCartesianGaussians)
    test.angmom = 1
    assert isinstance(test._angmom, int) and test._angmom == 1
    test.angmom = 0
    assert isinstance(test._angmom, int) and test._angmom == 0
    with pytest.raises(ValueError):
        test.angmom = -2
    with pytest.raises(TypeError):
        test.angmom = "0"
    with pytest.raises(TypeError):
        test.angmom = 0.0
    with pytest.raises(TypeError):
        test.angmom = None


def test_angmom_getter():
    """Test getter for ContractedCartesianGaussians.angmom."""
    test = skip_init(ContractedCartesianGaussians)
    test._angmom = 1
    assert test.angmom == 1


def test_exps_setter():
    """Test setter for ContractedCartesianGaussians.exps."""
    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    with pytest.raises(TypeError):
        test.exps = [1, 2, 3]
    with pytest.raises(TypeError):
        test.exps = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.coeffs = np.array([1.0, 2.0, 3.0])
        test.exps = np.array([4.0, 5.0])


def test_exps_getter():
    """Test getter for ContractedCartesianGaussians.exps."""
    test = skip_init(ContractedCartesianGaussians)
    test._exps = [2.0, 3.0]
    assert test.exps == [2.0, 3.0]


def test_coeffs_setter():
    """Test setter for ContractedCartesianGaussians.coeffs."""
    test = skip_init(ContractedCartesianGaussians)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    with pytest.raises(TypeError):
        test.coeffs = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coeffs = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0])
        test.coeffs = np.array([1.0, 2.0, 3.0])


def test_coeffs_getter():
    """Test getter for ContractedCartesianGaussians.coeffs."""
    test = skip_init(ContractedCartesianGaussians)
    test._coeffs = [2.0, 3.0]
    assert test.coeffs == [2.0, 3.0]


def tests_init():
    """Test ContractedCartesianGaussians.__init__."""
    test = ContractedCartesianGaussians(
        1,
        np.array([0, 1, 2]),
        0,
        np.array([1, 2, 3, 4], dtype=float),
        np.array([5, 6, 7, 8], dtype=float),
    )
    assert test._angmom == 1
    assert np.allclose(test._coord, np.array([0, 1, 2]))
    assert test._charge == 0
    assert np.allclose(test._coeffs, np.array([1, 2, 3, 4]))
    assert np.allclose(test._exps, np.array([5, 6, 7, 8]))


def test_angmom_components():
    """Test ContractedCartesianGaussians.angmom_components."""
    test = skip_init(ContractedCartesianGaussians)
    test._angmom = 0
    assert test.angmom_components == [(0, 0, 0)]
    test._angmom = 1
    assert test.angmom_components == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    test._angmom = 2
    assert test.angmom_components == [
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
    ]
    test._angmom = 3
    assert test.angmom_components == [
        (0, 0, 3),
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 0),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
        (3, 0, 0),
    ]
    test._angmom = 10
    assert len(test.angmom_components) == 11 * 12 / 2


# TODO: Test norm using actual integrals
def test_cartesian_gaussian_norm():
    """Test cartesian_gaussian_norm."""
    assert np.isclose(
        [cartesian_gaussian_norm(np.array([0, 0, 0]), 0.25)], [0.2519794355383807303479140]
    )
    assert np.isclose(
        [cartesian_gaussian_norm(np.array([2, 0, 1]), 0.5)], [0.6920252830162908851679097]
    )
