"""Test gbasis.integrals.boys_functions module.

Tests for standard Coulomb Boys function (Week 1) and
erf/erfc-attenuated potentials (Week 5).
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp1f1

from gbasis.integrals.boys_functions import (
    boys_function_all_orders,
    boys_function_erf,
    boys_function_erfc,
    boys_function_standard,
    get_boys_function,
    boys_function_recursion,
    boys_function_mpmath,
)


try:
    import mpmath  # noqa: F401
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


class TestBoysStandard:
    """Tests for the standard Boys function (Coulomb potential)."""

    def test_boys_m0_zero(self):
        """Test F_0(0) = 1."""
        result = boys_function_standard(np.array([0]), np.array([0.0]))
        assert np.allclose(result, 1.0)

    def test_boys_m0_limit(self):
        """Test F_0(T) -> sqrt(pi)/(2*sqrt(T)) for large T."""
        T = np.array([100.0])
        result = boys_function_standard(np.array([0]), T)
        expected = np.sqrt(np.pi) / (2 * np.sqrt(T))
        assert np.allclose(result, expected, rtol=0.01)

    def test_boys_m1_zero(self):
        """Test F_1(0) = 1/3."""
        result = boys_function_standard(np.array([1]), np.array([0.0]))
        assert np.allclose(result, 1.0 / 3.0)

    def test_boys_general_zero(self):
        """Test F_m(0) = 1/(2m+1)."""
        for m in range(10):
            result = boys_function_standard(np.array([m]), np.array([0.0]))
            expected = 1.0 / (2 * m + 1)
            assert np.allclose(result, expected), f"Failed for m={m}"

    def test_boys_consistency_with_hyp1f1(self):
        """Test that our implementation matches scipy's hyp1f1."""
        orders = np.arange(5)
        T_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

        for m in orders:
            for T in T_values:
                result = boys_function_standard(np.array([m]), np.array([T]))
                expected = hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1)
                assert np.allclose(result, expected), f"Failed for m={m}, T={T}"

    def test_boys_broadcasting(self):
        """Test that Boys function handles broadcasting correctly (shape + values)."""
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        result = boys_function_standard(orders, T)
        assert result.shape == (3, 3)

        # Verify each element matches individual scalar calls
        for i, m in enumerate([0, 1, 2]):
            for j, t in enumerate([0.5, 1.0, 2.0]):
                expected = boys_function_standard(np.array([m]), np.array([t]))[0]
                assert np.isclose(result[i, j], expected), \
                    f"Broadcasting value mismatch at m={m}, T={t}"

    def test_boys_recurrence(self):
        """Test the recurrence relation: (2m+1)*F_m(T) = 2T*F_{m+1}(T) + exp(-T)."""
        T = np.array([1.0, 2.0, 3.0])
        for m in range(5):
            fm = boys_function_standard(np.array([m]), T)
            fm1 = boys_function_standard(np.array([m + 1]), T)
            lhs = (2 * m + 1) * fm
            rhs = 2 * T * fm1 + np.exp(-T)
            assert np.allclose(lhs, rhs), f"Recurrence failed for m={m}"


class TestBoysAllOrders:
    """Tests for the all-orders Boys function."""

    def test_matches_standard_small_T(self):
        """Test that all-orders function matches standard for small T."""
        m_max = 5
        T = np.array([0.1, 0.5, 1.0, 2.0])
        result = boys_function_all_orders(m_max, T)

        for m in range(m_max + 1):
            expected = boys_function_standard(np.array([m]), T)
            assert np.allclose(result[m], expected), f"Failed for m={m}"

    def test_matches_standard_large_T(self):
        """Test that all-orders function matches standard for large T (>= 25)."""
        m_max = 5
        T = np.array([30.0, 50.0, 100.0])
        result = boys_function_all_orders(m_max, T)

        for m in range(m_max + 1):
            expected = boys_function_standard(np.array([m]), T)
            np.testing.assert_allclose(result[m], expected.flatten(), rtol=1e-8,
                err_msg=f"All-orders doesn't match standard for m={m}, large T")

    def test_output_shape(self):
        """Test that all-orders function returns correct shape (m_max+1, *T.shape)."""
        m_max = 4
        T = np.array([0.5, 1.0, 2.0, 5.0, 30.0])
        result = boys_function_all_orders(m_max, T)
        assert result.shape == (m_max + 1, T.shape[0]), \
            f"Expected shape {(m_max + 1, T.shape[0])}, got {result.shape}"

    def test_small_T_series_branch(self):
        """Exercise the small-T series path and compare with reference."""
        m_max = 6
        T = np.array([0.0, 1e-12, 1e-9])
        result = boys_function_all_orders(m_max, T)

        for m in range(m_max + 1):
            expected = boys_function_standard(np.array([m]), T)
            np.testing.assert_allclose(
                result[m], expected,
                rtol=1e-12, atol=1e-14,
                err_msg=f"Small-T mismatch at m={m}")

    def test_multidim_broadcasting(self):
        """Check broadcasting for multidimensional T arrays."""
        m_max = 3
        T = np.array([[0.2, 1.0, 5.0], [0.4, 2.0, 10.0]])[:, None, :]
        result = boys_function_all_orders(m_max, T)

        assert result.shape == (m_max + 1, *T.shape)

        # Verify values against scalar reference
        for idx, t_val in np.ndenumerate(T):
            for m in range(m_max + 1):
                expected = boys_function_standard(np.array([m]), np.array([t_val]))[0]
                assert np.isclose(result[m][idx], expected), f"Mismatch at m={m}, idx={idx}"

    def test_recursion_identity(self):
        """Verify recurrence holds on all-orders output."""
        m_max = 8
        T = np.array([0.3, 1.5, 4.0])
        all_vals = boys_function_all_orders(m_max, T)

        for m in range(m_max):
            lhs = (2 * m + 1) * all_vals[m]
            rhs = 2 * T * all_vals[m + 1] + np.exp(-T)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-14,
                err_msg=f"Recurrence failed for m={m}")

    def test_random_values_against_reference(self):
        """Random spot-check across orders and T values."""
        rng = np.random.default_rng(0)
        T = 10 ** rng.uniform(-10, 4, size=20)
        m_max = 10
        result = boys_function_all_orders(m_max, T)

        for m in range(m_max + 1):
            expected = boys_function_standard(np.array([m]), T)
            np.testing.assert_allclose(result[m], expected, rtol=1e-11, atol=1e-14,
                err_msg=f"Random check mismatch at m={m}")


class TestGetBoysFunction:
    """Tests for the get_boys_function factory."""

    def test_coulomb_potential(self):
        """Test that Coulomb potential returns standard Boys function."""
        boys = get_boys_function("coulomb")
        result = boys(np.array([0]), np.array([1.0]))
        expected = boys_function_standard(np.array([0]), np.array([1.0]))
        assert np.allclose(result, expected)

    def test_aliases(self):
        """Test that aliases work correctly."""
        for alias in ["coulomb", "standard", "1/r"]:
            boys = get_boys_function(alias)
            result = boys(np.array([0]), np.array([1.0]))
            assert result is not None

    def test_unknown_potential(self):
        """Test that unknown potential raises error."""
        with pytest.raises(ValueError):
            get_boys_function("unknown_potential")

    def test_erf_potential(self):
        """Test that erf potential returns correct function."""
        boys = get_boys_function("erf", omega=0.5)
        result = boys(np.array([0]), np.array([1.0]), rho=0.8)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_erfc_potential(self):
        """Test that erfc potential returns correct function."""
        boys = get_boys_function("erfc", omega=0.5)
        result = boys(np.array([0]), np.array([1.0]), rho=0.8)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_erf_aliases(self):
        """Test that erf aliases work correctly."""
        for alias in ["erf", "erf_coulomb"]:
            boys = get_boys_function(alias, omega=0.5)
            result = boys(np.array([0]), np.array([1.0]), rho=0.8)
            assert result is not None

    def test_erfc_aliases(self):
        """Test that erfc aliases work correctly."""
        for alias in ["erfc", "erfc_coulomb"]:
            boys = get_boys_function(alias, omega=0.5)
            result = boys(np.array([0]), np.array([1.0]), rho=0.8)
            assert result is not None

    def test_erf_requires_omega(self):
        """Test that erf potential requires omega parameter."""
        with pytest.raises(ValueError):
            get_boys_function("erf")

    def test_erfc_requires_omega(self):
        """Test that erfc potential requires omega parameter."""
        with pytest.raises(ValueError):
            get_boys_function("erfc")

    def test_erf_requires_rho(self):
        """Test that erf Boys function requires rho at call time."""
        boys = get_boys_function("erf", omega=0.5)
        with pytest.raises(ValueError):
            boys(np.array([0]), np.array([1.0]))

    def test_erfc_requires_rho(self):
        """Test that erfc Boys function requires rho at call time."""
        boys = get_boys_function("erfc", omega=0.5)
        with pytest.raises(ValueError):
            boys(np.array([0]), np.array([1.0]))

    def test_case_insensitive(self):
        """Test that potential matching is case-insensitive."""
        for alias in ["COULOMB", "Coulomb", "Standard", "1/R"]:
            boys = get_boys_function(alias)
            result = boys(np.array([0]), np.array([1.0]))
            expected = boys_function_standard(np.array([0]), np.array([1.0]))
            assert np.allclose(result, expected), f"Failed for '{alias}'"


class TestBoysNumericalIntegration:
    """Compare Boys function with numerical integration."""

    def test_numerical_integration_m0(self):
        """Test F_0(T) against numerical integration."""

        def boys_integrand(t, T, m):
            return t ** (2 * m) * np.exp(-T * t ** 2)

        for T in [0.5, 1.0, 2.0, 5.0]:
            result_analytic = boys_function_standard(np.array([0]), np.array([T]))[0]
            result_numeric, _ = quad(boys_integrand, 0, 1, args=(T, 0))
            assert np.allclose(result_analytic, result_numeric, rtol=1e-6)

    def test_numerical_integration_m2(self):
        """Test F_2(T) against numerical integration."""

        def boys_integrand(t, T, m):
            return t ** (2 * m) * np.exp(-T * t ** 2)

        for T in [0.5, 1.0, 2.0]:
            result_analytic = boys_function_standard(np.array([2]), np.array([T]))[0]
            result_numeric, _ = quad(boys_integrand, 0, 1, args=(T, 2))
            assert np.allclose(result_analytic, result_numeric, rtol=1e-6)
class TestBoysErf:
    """Tests for the erf-attenuated Boys function."""

    def test_erf_large_omega_recovers_coulomb(self):
        """Test that erf with large omega approaches standard Coulomb.

        As omega -> infinity, erf(omega*r)/r -> 1/r.
        """
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        rho = np.ones_like(T) * 0.8

        result_erf = boys_function_erf(orders, T, rho, omega=1000.0)
        result_std = boys_function_standard(orders, T)

        np.testing.assert_allclose(result_erf, result_std, rtol=1e-5,
            err_msg="erf with large omega should recover standard Coulomb")

    def test_erf_small_omega_approaches_zero(self):
        """Test that erf with small omega approaches zero.

        As omega -> 0, erf(omega*r)/r -> 0.
        """
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        rho = np.ones_like(T) * 0.8

        result_erf = boys_function_erf(orders, T, rho, omega=0.001)
        result_std = boys_function_standard(orders, T)
        # With omega=0.001, erf values should be much smaller than Coulomb
        ratio = np.abs(result_erf) / (np.abs(result_std) + 1e-30)
        assert np.all(ratio < 0.01), "erf with small omega should be << Coulomb"

    def test_erf_positive_values(self):
        """Test that erf Boys function produces positive values."""
        orders = np.array([0, 1, 2, 3])[:, None]
        T = np.array([0.1, 0.5, 1.0, 5.0])[None, :]
        rho = np.ones_like(T) * 1.0

        result = boys_function_erf(orders, T, rho, omega=0.5)
        assert np.all(result >= 0), "erf Boys function should be non-negative"

    def test_erf_less_than_coulomb(self):
        """Test that erf Boys < standard Coulomb (since erf(x) <= 1)."""
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        rho = np.ones_like(T) * 0.8

        result_erf = boys_function_erf(orders, T, rho, omega=0.5)
        result_std = boys_function_standard(orders, T)

        assert np.all(result_erf <= result_std + 1e-15), \
            "erf Boys should be <= standard Coulomb"

    def test_erf_scaling_formula(self):
        """Test the erf scaling formula: scaling^(m+0.5) * F_m(scaling * T).

        Tests m=0,1,2 to ensure the exponent (m+0.5) is not hardcoded.
        """
        T = np.array([1.0])
        rho = np.array([0.8])
        omega = 0.5
        scaling = omega**2 / (omega**2 + rho)
        T_mod = scaling * T

        for m_val in [0, 1, 2]:
            m = np.array([m_val])
            expected = scaling ** (m_val + 0.5) * boys_function_standard(m, T_mod)
            result = boys_function_erf(m, T, rho, omega)
            np.testing.assert_allclose(result, expected, rtol=1e-12,
                err_msg=f"erf scaling formula failed for m={m_val}")


class TestBoysErfc:
    """Tests for the erfc-attenuated Boys function."""

    def test_erfc_small_omega_recovers_coulomb(self):
        """Test that erfc with small omega approaches standard Coulomb.

        As omega -> 0, erfc(omega*r)/r -> 1/r.
        """
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        rho = np.ones_like(T) * 0.8

        result_erfc = boys_function_erfc(orders, T, rho, omega=0.001)
        result_std = boys_function_standard(orders, T)

        np.testing.assert_allclose(result_erfc, result_std, rtol=0.01,
            err_msg="erfc with small omega should recover standard Coulomb")

    def test_erfc_large_omega_approaches_zero(self):
        """Test that erfc with large omega approaches zero.

        As omega -> infinity, erfc(omega*r)/r -> 0.
        """
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.5, 1.0, 2.0])[None, :]
        rho = np.ones_like(T) * 0.8

        result_erfc = boys_function_erfc(orders, T, rho, omega=1000.0)
        assert np.all(np.abs(result_erfc) < 1e-6), \
            "erfc with large omega should be near zero"

    def test_erf_plus_erfc_equals_coulomb(self):
        """Test that erf + erfc = Coulomb for any omega.

        This is the fundamental identity: erf(x) + erfc(x) = 1, so
        erf(omega*r)/r + erfc(omega*r)/r = 1/r.
        """
        orders = np.array([0, 1, 2, 3])[:, None]
        T = np.array([0.1, 0.5, 1.0, 2.0, 5.0])[None, :]
        rho = np.ones_like(T) * 1.2

        for omega in [0.1, 0.5, 1.0, 2.0]:
            result_erf = boys_function_erf(orders, T, rho, omega)
            result_erfc = boys_function_erfc(orders, T, rho, omega)
            result_std = boys_function_standard(orders, T)

            np.testing.assert_allclose(
                result_erf + result_erfc, result_std, rtol=1e-12,
                err_msg=f"erf + erfc != Coulomb for omega={omega}"
            )

    def test_erfc_positive_values(self):
        """Test that erfc Boys function produces positive values."""
        orders = np.array([0, 1, 2])[:, None]
        T = np.array([0.1, 0.5, 1.0, 5.0])[None, :]
        rho = np.ones_like(T) * 1.0

        result = boys_function_erfc(orders, T, rho, omega=0.5)
        assert np.all(result >= -1e-15), "erfc Boys function should be non-negative"


class TestBoysRecursion:
    """Tests for the downward recursion Boys function (Eq. 71)."""

    def test_recursion_matches_all_orders(self):
        """Test that recursion matches hyp1f1-based all_orders for various T."""
        m_max = 10
        T = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
        result_rec = boys_function_recursion(m_max, T)
        result_ref = boys_function_all_orders(m_max, T)

        np.testing.assert_allclose(result_rec, result_ref, rtol=1e-12, atol=1e-15,
            err_msg="Recursion doesn't match all_orders")

    def test_recursion_T_zero(self):
        """Test F_m(0) = 1/(2m+1) via recursion."""
        m_max = 15
        result = boys_function_recursion(m_max, np.array([0.0]))
        for m in range(m_max + 1):
            expected = 1.0 / (2 * m + 1)
            np.testing.assert_allclose(result[m], expected, rtol=1e-14,
                err_msg=f"F_{m}(0) incorrect")

    def test_recursion_large_T(self):
        """Test stability for large T values."""
        m_max = 10
        T = np.array([50.0, 100.0, 500.0])
        result_rec = boys_function_recursion(m_max, T)
        result_ref = boys_function_all_orders(m_max, T)

        np.testing.assert_allclose(result_rec, result_ref, rtol=1e-10,
            err_msg="Recursion unstable for large T")

    def test_recursion_high_order(self):
        """Test with m_max = 20 (high angular momentum)."""
        m_max = 20
        T = np.array([0.5, 2.0, 10.0])
        result_rec = boys_function_recursion(m_max, T)
        result_ref = boys_function_all_orders(m_max, T)

        np.testing.assert_allclose(result_rec, result_ref, rtol=1e-11,
            err_msg="Recursion fails at high order")

    def test_recursion_output_shape(self):
        """Test output shape matches convention."""
        m_max = 5
        T = np.array([0.5, 1.0, 2.0])
        result = boys_function_recursion(m_max, T)
        assert result.shape == (m_max + 1, 3)

    def test_recursion_multidim(self):
        """Test with multidimensional T array."""
        m_max = 4
        T = np.array([[0.1, 1.0], [5.0, 10.0]])
        result_rec = boys_function_recursion(m_max, T)
        result_ref = boys_function_all_orders(m_max, T)

        assert result_rec.shape == (m_max + 1, 2, 2)
        np.testing.assert_allclose(result_rec, result_ref, rtol=1e-12)

    def test_recursion_recurrence_identity(self):
        """Verify recurrence (2m+1)*F_m = 2T*F_{m+1} + exp(-T) holds."""
        m_max = 8
        T = np.array([0.3, 1.5, 4.0])
        all_vals = boys_function_recursion(m_max, T)

        for m in range(m_max):
            lhs = (2 * m + 1) * all_vals[m]
            rhs = 2 * T * all_vals[m + 1] + np.exp(-T)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-14,
                err_msg=f"Recurrence failed for m={m}")


@pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not installed")
class TestBoysMpmath:
    """Tests for the mpmath high-precision Boys function."""

    def test_mpmath_matches_standard(self):
        """Test that mpmath matches standard hyp1f1 for normal T values."""
        m_max = 5
        T = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        result_mp = boys_function_mpmath(m_max, T, dps=30)
        result_std = boys_function_all_orders(m_max, T)

        np.testing.assert_allclose(result_mp, result_std, rtol=1e-14,
            err_msg="mpmath doesn't match standard for normal T")

    def test_mpmath_output_shape(self):
        """Test that mpmath returns correct shape."""
        m_max = 3
        T = np.array([[0.5, 1.0], [2.0, 5.0]])
        result = boys_function_mpmath(m_max, T, dps=20)
        assert result.shape == (m_max + 1, 2, 2)

    def test_mpmath_T_zero(self):
        """Test F_m(0) = 1/(2m+1) via mpmath."""
        m_max = 10
        result = boys_function_mpmath(m_max, np.array([0.0]), dps=50)
        for m in range(m_max + 1):
            expected = 1.0 / (2 * m + 1)
            np.testing.assert_allclose(result[m], expected, rtol=1e-15)

    def test_mpmath_high_order(self):
        """Test mpmath accuracy for high orders (m=15+)."""
        m_max = 15
        T = np.array([1.0, 5.0])
        result_mp = boys_function_mpmath(m_max, T, dps=50)
        result_std = boys_function_all_orders(m_max, T)

        np.testing.assert_allclose(result_mp, result_std, rtol=1e-14,
            err_msg="mpmath high-order mismatch")

