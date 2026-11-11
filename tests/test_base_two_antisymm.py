"""Test gbasis.base_two_antisymm."""
import numpy as np
import pytest
from utils import disable_abstract, skip_init

from gbasis.base_two_antisymm import BaseTwoIndexAntiSymmetric
from gbasis.contractions import GeneralizedContractionShell


def test_init():
    """Test BaseTwoIndexAntiSymmetric.__init__."""
    Test = disable_abstract(BaseTwoIndexAntiSymmetric)
    test = skip_init(Test)
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    Test.__init__(test, [contractions])
    assert test._axes_contractions[0][0] == contractions
    with pytest.raises(TypeError):
        Test.__init__(test, [contractions], [contractions])


def test_contractions():
    """Test BaseTwoIndexAntiSymmetric.contractions."""
    Test = disable_abstract(BaseTwoIndexAntiSymmetric)
    cont = GeneralizedContractionShell(1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical")
    test = Test([cont])
    assert test.contractions[0] == cont


def test_construct_array_contraction():
    """Test BaseTwoIndexAntiSymmetric.construct_array_contraction abstract method."""
    Test = disable_abstract(
        BaseTwoIndexAntiSymmetric,
        dict_overwrite={
            "construct_array_contraction": BaseTwoIndexAntiSymmetric.construct_array_contraction
        },
    )
    contractions = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    with pytest.raises(TypeError):
        Test([contractions])


def test_construct_array_cartesian():
    """Test BaseTwoIndexAntiSymmetric.construct_array_cartesian."""
    cont_one = GeneralizedContractionShell(
        1, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 2, 3]), np.ones(1), np.ones(1), "spherical"
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))

    # Mock contraction function that returns an anti-symmetric intra-shell block for diagonal
    # and a general matrix for off-diagonal blocks
    def mock_contraction(self, c1, c2, scale=1.0):
        if c1 == c2:
            n = c1.num_cart
            mat = np.arange(n * n, dtype=float).reshape(n, n)
            mat = mat - mat.T
            return (mat * scale).reshape(1, n, 1, n)
        n1 = c1.num_cart
        n2 = c2.num_cart
        mat = np.arange(n1 * n2, dtype=float).reshape(n1, n2)
        return (mat * scale).reshape(1, n1, 1, n2)

    Test = disable_abstract(
        BaseTwoIndexAntiSymmetric,
        dict_overwrite={"construct_array_contraction": mock_contraction},
    )

    test = Test([cont_one, cont_two])
    arr = test.construct_array_cartesian(scale=2.0)

    # Verify overall anti-symmetry: A^T == -A
    assert np.allclose(arr, -arr.T)

    # Verify diagonal block 0,0 is untouched and matches intra-shell evaluation
    diag_0 = mock_contraction(test, cont_one, cont_one, scale=2.0).reshape(3, 3)
    assert np.allclose(arr[:3, :3], diag_0)

    # Verify off-diagonal block (1, 0) is -block(0, 1)^T
    assert np.allclose(arr[3:, :3], -arr[:3, 3:].T)

    # Test error on invalid keyword
    with pytest.raises(TypeError):
        test.construct_array_cartesian(invalid_kw=42)


def test_construct_array_spherical():
    """Test BaseTwoIndexAntiSymmetric.construct_array_spherical."""
    cont_one = GeneralizedContractionShell(
        1, np.array([0, 0, 0]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 0, 0]), np.ones(1), np.ones(1), "spherical"
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))

    def mock_contraction(self, c1, c2):
        if c1 == c2:
            n = c1.num_cart
            mat = np.arange(n * n, dtype=float).reshape(n, n)
            mat = mat - mat.T
            return mat.reshape(1, n, 1, n)
        n1 = c1.num_cart
        n2 = c2.num_cart
        mat = np.arange(n1 * n2, dtype=float).reshape(n1, n2)
        return mat.reshape(1, n1, 1, n2)

    Test = disable_abstract(
        BaseTwoIndexAntiSymmetric,
        dict_overwrite={"construct_array_contraction": mock_contraction},
    )

    test = Test([cont_one, cont_two])
    arr_sph = test.construct_array_spherical()

    # Spherical array must be strictly anti-symmetric: M^T == -M
    assert np.allclose(arr_sph, -arr_sph.T)


def test_construct_array_mix():
    """Test BaseTwoIndexAntiSymmetric.construct_array_mix."""
    cont_one = GeneralizedContractionShell(
        1, np.array([0, 0, 0]), np.ones(1), np.ones(1), "spherical"
    )
    cont_two = GeneralizedContractionShell(
        2, np.array([1, 0, 0]), np.ones(1), np.ones(1), "spherical"
    )
    cont_one.norm_cont = np.ones((1, cont_one.num_cart))
    cont_two.norm_cont = np.ones((1, cont_two.num_cart))

    def mock_contraction(self, c1, c2):
        if c1 == c2:
            n = c1.num_cart
            mat = np.arange(n * n, dtype=float).reshape(n, n)
            mat = mat - mat.T
            return mat.reshape(1, n, 1, n)
        n1 = c1.num_cart
        n2 = c2.num_cart
        mat = np.arange(n1 * n2, dtype=float).reshape(n1, n2)
        return mat.reshape(1, n1, 1, n2)

    Test = disable_abstract(
        BaseTwoIndexAntiSymmetric,
        dict_overwrite={"construct_array_contraction": mock_contraction},
    )

    test = Test([cont_one, cont_two])

    # Test input validation
    with pytest.raises(TypeError):
        test.construct_array_mix("cartesian")
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian", "invalid"])
    with pytest.raises(ValueError):
        test.construct_array_mix(["cartesian"])  # length mismatch

    arr_mix = test.construct_array_mix(["cartesian", "spherical"])
    # Off-diagonal blocks in mixed representation should satisfy anti-symmetry
    n_cart = cont_one.num_cart  # 3
    block_01 = arr_mix[:n_cart, n_cart:]
    block_10 = arr_mix[n_cart:, :n_cart]
    assert np.allclose(block_10, -block_01.T)


def test_construct_array_lincomb():
    """Test BaseTwoIndexAntiSymmetric.construct_array_lincomb."""
    cont = GeneralizedContractionShell(
        1, np.array([0, 0, 0]), np.ones(1), np.ones(1), "spherical"
    )
    cont.norm_cont = np.ones((1, cont.num_cart))

    def mock_contraction(self, c1, c2):
        n = c1.num_cart
        mat = np.arange(n * n, dtype=float).reshape(n, n)
        mat = mat - mat.T
        return mat.reshape(1, n, 1, n)

    Test = disable_abstract(
        BaseTwoIndexAntiSymmetric,
        dict_overwrite={"construct_array_contraction": mock_contraction},
    )
    test = Test([cont])
    transform = np.array([[1.0, -0.5, 0.2], [0.3, 0.8, -0.1]])
    arr_mo = test.construct_array_lincomb(transform, ["cartesian"])

    # Anti-symmetry should be preserved under linear combination: (T A T^T)^T = - (T A T^T)
    assert np.allclose(arr_mo, -arr_mo.T)
