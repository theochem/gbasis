"""Test gbasis.contractions."""
from gbasis.contractions import GeneralizedContractionShell
import numpy as np
import pytest
from utils import skip_init


def test_coord_setter():
    """Test setter for GeneralizedContractionShell.coord."""
    test = skip_init(GeneralizedContractionShell)
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
    """Test getter for GeneralizedContractionShell.coord."""
    test = skip_init(GeneralizedContractionShell)
    test._coord = 2
    assert test.coord == 2


def test_angmom_setter():
    """Test setter for GeneralizedContractionShell.angmom."""
    test = skip_init(GeneralizedContractionShell)
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
    """Test getter for GeneralizedContractionShell.angmom."""
    test = skip_init(GeneralizedContractionShell)
    test._angmom = 1
    assert test.angmom == 1


def test_exps_setter():
    """Test setter for GeneralizedContractionShell.exps."""
    test = skip_init(GeneralizedContractionShell)
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(GeneralizedContractionShell)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(GeneralizedContractionShell)
    with pytest.raises(TypeError):
        test.exps = [1, 2, 3]
    with pytest.raises(TypeError):
        test.exps = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.coeffs = np.array([1.0, 2.0, 3.0])
        test.exps = np.array([4.0, 5.0])
    with pytest.raises(ValueError):
        test.coeffs = np.array([[1.0], [2.0], [3.0]])
        test.exps = np.array([4.0, 5.0])


def test_exps_getter():
    """Test getter for GeneralizedContractionShell.exps."""
    test = skip_init(GeneralizedContractionShell)
    test._exps = [2.0, 3.0]
    assert test.exps == [2.0, 3.0]


def test_coeffs_setter():
    """Test setter for GeneralizedContractionShell.coeffs."""
    test = skip_init(GeneralizedContractionShell)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(GeneralizedContractionShell)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(GeneralizedContractionShell)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([[1.0], [2.0], [3.0]])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(GeneralizedContractionShell)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1, 4], [2, 5], [3, 6]]))
    )

    test = skip_init(GeneralizedContractionShell)
    with pytest.raises(TypeError):
        test.coeffs = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coeffs = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0])
        test.coeffs = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0, 6.0])
        test.coeffs = np.array([[[1.0, 2.0, 3.0]]])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0, 6.0])
        test.coeffs = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0])
        test.coeffs = np.array([[1.0], [2.0], [3.0]])


def test_coeffs_getter():
    """Test getter for GeneralizedContractionShell.coeffs."""
    test = skip_init(GeneralizedContractionShell)
    test._coeffs = [2.0, 3.0]
    assert test.coeffs == [2.0, 3.0]


def tests_init():
    """Test GeneralizedContractionShell.__init__."""
    test = GeneralizedContractionShell(
        1,
        np.array([0, 1, 2]),
        np.array([1, 2, 3, 4], dtype=float),
        np.array([5, 6, 7, 8], dtype=float),
        "spherical",
    )
    assert test._angmom == 1
    assert np.allclose(test._coord, np.array([0, 1, 2]))
    assert np.allclose(test._coeffs, np.array([[1], [2], [3], [4]]))
    assert np.allclose(test._exps, np.array([5, 6, 7, 8]))


def test_angmom_components_cart():
    """Test GeneralizedContractionShell.angmom_components_cart."""
    test = skip_init(GeneralizedContractionShell)
    test._angmom = 0
    assert np.allclose(test.angmom_components_cart, [(0, 0, 0)])
    test._angmom = 1
    assert np.allclose(test.angmom_components_cart, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    test._angmom = 2
    assert np.allclose(
        test.angmom_components_cart,
        [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
    )
    test._angmom = 3
    assert np.allclose(
        test.angmom_components_cart,
        [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ],
    )
    test._angmom = 10
    assert len(test.angmom_components_cart) == 11 * 12 / 2


def test_angmom_components_sph():
    """Test GeneralizedContractionShell.angmom_components_sph."""
    test = skip_init(GeneralizedContractionShell)
    test._angmom = 0
    assert test.angmom_components_sph == ("c0",)
    test._angmom = 1
    assert test.angmom_components_sph == ("c1", "s1", "c0")
    test._angmom = 2
    assert test.angmom_components_sph == ("s2", "s1", "c0", "c1", "c2")
    test._angmom = 3
    assert test.angmom_components_sph == ("s3", "s2", "s1", "c0", "c1", "c2", "c3")


# TODO: Test norm using actual integrals
# TODO: add more tests
def test_norm_prim_cart():
    """Test GeneralizedContractionShell.norm_prim_cart."""
    test = GeneralizedContractionShell(
        0, np.array([0, 0, 0]), np.array([1.0]), np.array([0.25]), "spherical"
    )
    assert np.isclose(test.norm_prim_cart, 0.2519794355383807303479140)
    test = GeneralizedContractionShell(
        3, np.array([0, 0, 0]), np.array([1.0]), np.array([0.5]), "spherical"
    )
    assert np.isclose(test.norm_prim_cart[7], 0.6920252830162908851679097)


def test_num_cart():
    """Test GeneralizedContractionShell.num_cart."""
    test = skip_init(GeneralizedContractionShell)
    last_num_cart = 0
    for i in range(100):
        test._angmom = i
        assert test.num_cart == last_num_cart + i + 1
        last_num_cart = test.num_cart


def test_num_sph():
    """Test GeneralizedContractionShell.num_sph."""
    test = skip_init(GeneralizedContractionShell)
    last_num_sph = 1
    for i in range(100):
        test._angmom = i
        assert test.num_sph == last_num_sph
        last_num_sph = test.num_sph + 2


def test_num_seg_cont():
    """Test GeneralizedContractionShell.num_seg_cont."""
    test = skip_init(GeneralizedContractionShell)
    test._coeffs = np.random.rand(10, 21)
    assert test.num_seg_cont == 21


def test_assign_norm_cont():
    """Test GeneralizedContractionShell.assign_norm_cont."""
    test = GeneralizedContractionShell(
        0, np.array([0, 0, 0]), np.array([1.0]), np.array([0.25]), "spherical"
    )
    test.assign_norm_cont()
    assert np.allclose(test.norm_cont, 1)

    test = GeneralizedContractionShell(
        1, np.array([0, 0, 0]), np.array([1.0]), np.array([0.25]), "spherical"
    )
    test.assign_norm_cont()
    assert np.allclose(test.norm_cont, 1)

    test = GeneralizedContractionShell(
        2, np.array([0, 0, 0]), np.array([1.0]), np.array([0.25]), "spherical"
    )
    test.assign_norm_cont()
    assert np.allclose(test.norm_cont, 1)
