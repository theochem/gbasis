"""Test gbasis.integrals._two_elec_int_improved module.

Tests for VRR (Week 2), ETR + contraction (Week 3), and HRR + full pipeline (Week 4).
"""

import numpy as np
from scipy.special import hyp1f1

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._schwarz_screening import SchwarzScreener
from gbasis.integrals._two_elec_int import (
    _compute_two_elec_integrals,
    _compute_two_elec_integrals_angmom_zero,
)
from gbasis.integrals._two_elec_int_improved import (
    _electron_transfer_recursion,
    _get_factorial2_norm,
    _optimized_contraction,
    _vertical_recursion_relation,
    compute_two_electron_integrals_os_hgp,
)
from gbasis.integrals.electron_repulsion import (
    ElectronRepulsionIntegralImproved,
    electron_repulsion_integral_improved,
)
from gbasis.integrals.point_charge import PointChargeIntegral


class TestVerticalRecursion:
    """Tests for the Vertical Recursion Relation (VRR)."""

    def test_vrr_base_case(self):
        """Test that VRR preserves base case [00|00]^m."""
        m_max = 4
        n_prim = 2

        # Create mock integrals_m (base case values)
        integrals_m = np.random.rand(m_max, n_prim, n_prim, n_prim, n_prim)

        # Mock coordinates and exponents
        rel_coord_a = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        coord_wac = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        harm_mean = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1
        exps_sum_one = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1

        result = _vertical_recursion_relation(
            integrals_m, m_max, rel_coord_a, coord_wac, harm_mean, exps_sum_one
        )

        # Base case should be preserved at [m, 0, 0, 0]
        assert np.allclose(result[:, 0, 0, 0, ...], integrals_m)

    def test_vrr_output_shape(self):
        """Test that VRR output has correct shape."""
        m_max = 5
        n_prim = 3

        integrals_m = np.zeros((m_max, n_prim, n_prim, n_prim, n_prim))
        rel_coord_a = np.zeros((3, n_prim, n_prim, n_prim, n_prim))
        coord_wac = np.zeros((3, n_prim, n_prim, n_prim, n_prim))
        harm_mean = np.ones((n_prim, n_prim, n_prim, n_prim))
        exps_sum_one = np.ones((n_prim, n_prim, n_prim, n_prim))

        result = _vertical_recursion_relation(
            integrals_m, m_max, rel_coord_a, coord_wac, harm_mean, exps_sum_one
        )

        expected_shape = (m_max, m_max, m_max, m_max, n_prim, n_prim, n_prim, n_prim)
        assert result.shape == expected_shape

    def test_vrr_p_orbital_manual(self):
        """Test VRR first recursion step for p-orbital manually.

        For p-orbital, VRR computes:
        [1,0|00]^0 = (P-A)_x * [00|00]^0 - (rho/zeta)*(Q-P)_x * [00|00]^1
        """
        m_max = 2
        integrals_m = np.array([[[[[1.0]]]], [[[[0.5]]]]])

        PA_x, PA_y, PA_z = 0.3, 0.0, 0.0
        PQ_x, PQ_y, PQ_z = 0.2, 0.0, 0.0
        rho = 0.6
        zeta = 1.5

        rel_coord_a = np.array([[[[[PA_x]]]], [[[[PA_y]]]], [[[[PA_z]]]]])
        coord_wac = np.array([[[[[PQ_x]]]], [[[[PQ_y]]]], [[[[PQ_z]]]]])
        harm_mean = np.array([[[[rho]]]])
        exps_sum_one = np.array([[[[zeta]]]])

        result = _vertical_recursion_relation(
            integrals_m, m_max, rel_coord_a, coord_wac, harm_mean, exps_sum_one
        )

        # Manual: [1,0,0|00]^0 = PA_x * [00|00]^0 - (rho/zeta)*PQ_x * [00|00]^1
        expected = PA_x * 1.0 - (rho / zeta) * PQ_x * 0.5
        assert np.allclose(result[0, 1, 0, 0, 0, 0, 0, 0], expected)

    def test_vrr_s_orbital_no_change(self):
        """Test VRR with s-orbital where no recursion is needed."""
        m_max = 1
        integrals_m = np.array([[[[[3.14]]]]])

        rel_coord_a = np.zeros((3, 1, 1, 1, 1))
        coord_wac = np.zeros((3, 1, 1, 1, 1))
        harm_mean = np.ones((1, 1, 1, 1))
        exps_sum_one = np.ones((1, 1, 1, 1))

        result = _vertical_recursion_relation(
            integrals_m, m_max, rel_coord_a, coord_wac, harm_mean, exps_sum_one
        )

        assert np.allclose(result[0, 0, 0, 0], 3.14)


class TestElectronTransferRecursion:
    """Tests for the Electron Transfer Recursion (ETR)."""

    def test_etr_base_case(self):
        """Test that ETR preserves base case (discards m, keeps a)."""
        m_max = 4
        m_max_c = 3
        n_prim = 2

        # Create mock VRR output
        integrals_vert = np.random.rand(m_max, m_max, m_max, m_max, n_prim, n_prim, n_prim, n_prim)

        rel_coord_c = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        rel_coord_a = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        exps_sum_one = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1
        exps_sum_two = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1

        result = _electron_transfer_recursion(
            integrals_vert, m_max, m_max_c, rel_coord_c, rel_coord_a, exps_sum_one, exps_sum_two
        )

        # Base case: [0,0,0, a_x, a_y, a_z] should equal integrals_vert[0, a_x, a_y, a_z]
        assert np.allclose(result[0, 0, 0, ...], integrals_vert[0, ...])

    def test_etr_output_shape(self):
        """Test that ETR output has correct shape."""
        m_max = 3
        m_max_c = 2
        n_prim = 2

        integrals_vert = np.random.rand(m_max, m_max, m_max, m_max, n_prim, n_prim, n_prim, n_prim)

        rel_coord_c = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        rel_coord_a = np.random.rand(3, n_prim, n_prim, n_prim, n_prim)
        exps_sum_one = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1
        exps_sum_two = np.random.rand(n_prim, n_prim, n_prim, n_prim) + 0.1

        result = _electron_transfer_recursion(
            integrals_vert, m_max, m_max_c, rel_coord_c, rel_coord_a, exps_sum_one, exps_sum_two
        )

        expected_shape = (
            m_max_c,
            m_max_c,
            m_max_c,
            m_max,
            m_max,
            m_max,
            n_prim,
            n_prim,
            n_prim,
            n_prim,
        )
        assert result.shape == expected_shape


class TestOptimizedContraction:
    """Tests for the optimized primitive contraction."""

    def test_output_shape(self):
        """Test that contraction output has correct shape."""
        K, M = 2, 3
        integrals_etransf = np.random.rand(1, 1, 1, 1, 1, 1, K, K, K, K)
        exps = np.random.rand(4, K) + 0.1
        coeffs = np.random.rand(4, K, M)
        angmoms = np.array([0, 0, 0, 0])

        result = _optimized_contraction(integrals_etransf, exps, coeffs, angmoms)

        expected_shape = (1, 1, 1, 1, 1, 1, M, M, M, M)
        assert result.shape == expected_shape

    def test_accepts_tuples(self):
        """Test that contraction accepts tuples as well as arrays."""
        K, M = 2, 2
        integrals_etransf = np.random.rand(1, 1, 1, 1, 1, 1, K, K, K, K)
        exps = tuple(np.random.rand(K) + 0.1 for _ in range(4))
        coeffs = tuple(np.random.rand(K, M) for _ in range(4))
        angmoms = (0, 0, 0, 0)

        result = _optimized_contraction(integrals_etransf, exps, coeffs, angmoms)

        expected_shape = (1, 1, 1, 1, 1, 1, M, M, M, M)
        assert result.shape == expected_shape


class TestFactorial2Norm:
    """Tests for the factorial2 normalization helper."""

    def test_s_orbital_norm(self):
        """Test normalization for s-orbital (L=0)."""
        s_key = ((0, 0, 0),)
        norm = _get_factorial2_norm(s_key)
        # (2*0-1)!! = (-1)!! = 1, so norm = 1/sqrt(1) = 1
        assert np.allclose(norm, 1.0)

    def test_p_orbital_norm(self):
        """Test normalization for p-orbital (L=1)."""
        p_key = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        norm = _get_factorial2_norm(p_key)
        # Each component has one (2*1-1)!! = 1!! = 1 and two (2*0-1)!! = 1
        # So norm = 1/sqrt(1*1*1) = 1 for all
        assert np.allclose(norm, 1.0)

    def test_caching(self):
        """Test that factorial2 normalization is cached."""
        d_key = ((2, 0, 0), (1, 1, 0))
        norm1 = _get_factorial2_norm(d_key)
        norm2 = _get_factorial2_norm(d_key)
        assert np.allclose(norm1, norm2)


class TestImprovedVsOld:
    """Compare improved OS+HGP implementation against old implementation."""

    def test_ssss_matches_old(self):
        """Test (ss|ss) integrals match old implementation.

        Note: The old _compute_two_elec_integrals has a known limitation with
        all-zero angular momentum, so we compare against the specialized
        _compute_two_elec_integrals_angmom_zero function instead.
        """
        boys_func = PointChargeIntegral.boys_func

        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        angmom_comp = np.array([[0, 0, 0]])

        exps_a = np.array([1.0, 0.5])
        exps_b = np.array([0.8])
        exps_c = np.array([1.2, 0.6])
        exps_d = np.array([0.9])

        coeffs_a = np.array([[1.0], [0.5]])
        coeffs_b = np.array([[1.0]])
        coeffs_c = np.array([[1.0], [0.3]])
        coeffs_d = np.array([[1.0]])

        # Old implementation (specialized for angmom zero)
        result_old = _compute_two_elec_integrals_angmom_zero(
            boys_func,
            coord_a,
            exps_a,
            coeffs_a,
            coord_b,
            exps_b,
            coeffs_b,
            coord_c,
            exps_c,
            coeffs_c,
            coord_d,
            exps_d,
            coeffs_d,
        )

        # New implementation
        result_new = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            angmom_comp,
            exps_a,
            coeffs_a,
            coord_b,
            0,
            angmom_comp,
            exps_b,
            coeffs_b,
            coord_c,
            0,
            angmom_comp,
            exps_c,
            coeffs_c,
            coord_d,
            0,
            angmom_comp,
            exps_d,
            coeffs_d,
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(ss|ss) integrals don't match between old and improved",
        )

    def test_spsp_matches_old(self):
        """Test (sp|sp) integrals match old implementation."""
        boys_func = PointChargeIntegral.boys_func

        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.5, 0.0, 0.0])
        coord_c = np.array([0.0, 1.5, 0.0])
        coord_d = np.array([1.5, 1.5, 0.0])

        exps_a = np.array([1.0, 0.5])
        exps_b = np.array([0.8])
        exps_c = np.array([1.2])
        exps_d = np.array([0.9])

        coeffs_a = np.array([[1.0], [0.5]])
        coeffs_b = np.array([[1.0]])
        coeffs_c = np.array([[1.0]])
        coeffs_d = np.array([[1.0]])

        # s-orbital components
        s_comp = np.array([[0, 0, 0]])
        # p-orbital components
        p_comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Old implementation
        result_old = _compute_two_elec_integrals(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps_a,
            coeffs_a,
            coord_b,
            1,
            p_comp,
            exps_b,
            coeffs_b,
            coord_c,
            0,
            s_comp,
            exps_c,
            coeffs_c,
            coord_d,
            1,
            p_comp,
            exps_d,
            coeffs_d,
        )

        # New implementation
        result_new = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps_a,
            coeffs_a,
            coord_b,
            1,
            p_comp,
            exps_b,
            coeffs_b,
            coord_c,
            0,
            s_comp,
            exps_c,
            coeffs_c,
            coord_d,
            1,
            p_comp,
            exps_d,
            coeffs_d,
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(sp|sp) integrals don't match between old and improved",
        )


class TestHighAngularMomentum:
    """Test that high angular momentum integrals don't produce NaN/Inf."""

    def test_dddd_no_overflow(self):
        """Test (dd|dd) integrals produce finite values."""
        boys_func = PointChargeIntegral.boys_func

        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        exps = np.array([1.0])
        coeffs = np.array([[1.0]])

        # d-orbital components (L=2): xx, xy, xz, yy, yz, zz
        d_comp = np.array([[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]])

        result = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            2,
            d_comp,
            exps,
            coeffs,
            coord_b,
            2,
            d_comp,
            exps,
            coeffs,
            coord_c,
            2,
            d_comp,
            exps,
            coeffs,
            coord_d,
            2,
            d_comp,
            exps,
            coeffs,
        )

        assert np.all(np.isfinite(result)), "d-orbital integrals contain NaN or Inf"
        assert result.shape == (6, 6, 6, 6, 1, 1, 1, 1)


class TestPrimitiveScreening:
    """Tests for primitive-level screening (Eq. 64)."""

    def test_screening_zero_threshold(self):
        """With threshold=0, results match unscreened exactly."""

        def boys_func(orders, weighted_dist, rho=None):
            return hyp1f1(orders + 0.5, orders + 1.5, -weighted_dist) / (2 * orders + 1)

        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        exps = np.array([1.0, 0.5])
        coeffs = np.array([[1.0], [1.0]])
        s_comp = np.array([[0, 0, 0]])
        p_comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result_no_screen = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            1,
            p_comp,
            exps,
            coeffs,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            1,
            p_comp,
            exps,
            coeffs,
        )

        result_screened = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            1,
            p_comp,
            exps,
            coeffs,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            1,
            p_comp,
            exps,
            coeffs,
            primitive_threshold=0.0,
        )

        np.testing.assert_array_equal(result_no_screen, result_screened)

    def test_screening_reasonable_threshold(self):
        """With reasonable threshold, results match within tolerance."""

        def boys_func(orders, weighted_dist, rho=None):
            return hyp1f1(orders + 0.5, orders + 1.5, -weighted_dist) / (2 * orders + 1)

        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        exps = np.array([1.0, 0.5, 0.1])
        coeffs = np.array([[1.0], [1.0], [1.0]])
        s_comp = np.array([[0, 0, 0]])

        result_no_screen = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            0,
            s_comp,
            exps,
            coeffs,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            0,
            s_comp,
            exps,
            coeffs,
        )

        result_screened = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            0,
            s_comp,
            exps,
            coeffs,
            coord_a,
            0,
            s_comp,
            exps,
            coeffs,
            coord_b,
            0,
            s_comp,
            exps,
            coeffs,
            primitive_threshold=1e-12,
        )

        np.testing.assert_allclose(result_no_screen, result_screened, atol=1e-10)


class TestContractionReordering:
    """Tests for contraction reordering (l_a >= l_b, l_c >= l_d, l_a >= l_c)."""

    def test_sp_ps_reordering(self):
        """Test that (sp|ps) gives correct results with bra/ket swapping."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        # Create shells: s (L=0) and p (L=1)
        shell_s_a = GeneralizedContractionShell(
            0, coord_a, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_p_b = GeneralizedContractionShell(
            1, coord_b, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_p_c = GeneralizedContractionShell(
            1, coord_c, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_s_d = GeneralizedContractionShell(
            0, coord_d, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )

        # (sp|ps) triggers bra_swapped and ket_swapped
        result = ElectronRepulsionIntegralImproved.construct_array_contraction(
            shell_s_a, shell_p_b, shell_p_c, shell_s_d
        )

        # Verify result is finite and has correct shape
        # Shape: (M_a, L_a, M_b, L_b, M_c, L_c, M_d, L_d)
        assert result.shape == (1, 1, 1, 3, 1, 3, 1, 1)
        assert np.all(np.isfinite(result)), "Reordered integrals contain NaN or Inf"

        # Compare with direct compute (no reordering) to verify correctness
        boys_func = PointChargeIntegral.boys_func

        s_comp = np.array([[0, 0, 0]])
        p_comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        direct = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            0,
            s_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_b,
            1,
            p_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_c,
            1,
            p_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_d,
            0,
            s_comp,
            np.array([1.0]),
            np.array([[1.0]]),
        )
        # direct shape: (L_a, L_b, L_c, L_d, M_a, M_b, M_c, M_d)
        direct_transposed = np.transpose(direct, (4, 0, 5, 1, 6, 2, 7, 3))

        np.testing.assert_allclose(
            result,
            direct_transposed,
            rtol=1e-10,
            err_msg="Reordered (sp|ps) doesn't match direct computation",
        )

    def test_pd_sp_braket_reordering(self):
        """Test (pd|sp) triggers bra-ket swap as well."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        shell_p_a = GeneralizedContractionShell(
            1, coord_a, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_d_b = GeneralizedContractionShell(
            2, coord_b, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_s_c = GeneralizedContractionShell(
            0, coord_c, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_p_d = GeneralizedContractionShell(
            1, coord_d, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )

        # (pd|sp): bra_swapped (p<d), ket_swapped (s<p), then braket_swapped (d>p, no swap)
        result = ElectronRepulsionIntegralImproved.construct_array_contraction(
            shell_p_a, shell_d_b, shell_s_c, shell_p_d
        )

        # Shape: (M_a, L_p, M_b, L_d, M_c, L_s, M_d, L_p)
        assert result.shape == (1, 3, 1, 6, 1, 1, 1, 3)
        assert np.all(np.isfinite(result)), "Reordered integrals contain NaN or Inf"

        # Compare with direct (no reordering)
        boys_func = PointChargeIntegral.boys_func

        s_comp = np.array([[0, 0, 0]])
        p_comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        d_comp = np.array([[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]])

        direct = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord_a,
            1,
            p_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_b,
            2,
            d_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_c,
            0,
            s_comp,
            np.array([1.0]),
            np.array([[1.0]]),
            coord_d,
            1,
            p_comp,
            np.array([1.0]),
            np.array([[1.0]]),
        )
        direct_transposed = np.transpose(direct, (4, 0, 5, 1, 6, 2, 7, 3))

        np.testing.assert_allclose(
            result,
            direct_transposed,
            rtol=1e-10,
            err_msg="Reordered (pd|sp) doesn't match direct computation",
        )


class TestSchwarzScreening:
    """Tests for Schwarz screening integration."""

    def test_schwarz_no_effect(self):
        """With threshold=0, results match unscreened exactly."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])

        shell_s = GeneralizedContractionShell(
            0, coord_a, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_p = GeneralizedContractionShell(
            1, coord_b, np.array([[1.0]]), np.array([0.8]), "cartesian"
        )
        basis = [shell_s, shell_p]

        result_no_screen = electron_repulsion_integral_improved(basis, notation="chemist")
        result_screened = electron_repulsion_integral_improved(
            basis, notation="chemist", schwarz_threshold=0.0
        )

        np.testing.assert_array_equal(result_no_screen, result_screened)

    def test_schwarz_screening_tight(self):
        """With tight threshold, results match within tolerance."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])

        shell_s1 = GeneralizedContractionShell(
            0, coord_a, np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_s2 = GeneralizedContractionShell(
            0, coord_b, np.array([[1.0]]), np.array([0.8]), "cartesian"
        )
        shell_s3 = GeneralizedContractionShell(
            0, coord_c, np.array([[1.0]]), np.array([0.6]), "cartesian"
        )
        basis = [shell_s1, shell_s2, shell_s3]

        result_no_screen = electron_repulsion_integral_improved(basis, notation="chemist")
        result_screened = electron_repulsion_integral_improved(
            basis, notation="chemist", schwarz_threshold=1e-12
        )

        np.testing.assert_allclose(result_no_screen, result_screened, atol=1e-10)

    def test_schwarz_statistics(self):
        """Verify screening statistics are tracked."""
        shell_s1 = GeneralizedContractionShell(
            0, np.array([0.0, 0.0, 0.0]), np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_s2 = GeneralizedContractionShell(
            0, np.array([100.0, 0.0, 0.0]), np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        shell_s3 = GeneralizedContractionShell(
            0, np.array([0.0, 100.0, 0.0]), np.array([[1.0]]), np.array([1.0]), "cartesian"
        )
        basis = [shell_s1, shell_s2, shell_s3]

        screener = SchwarzScreener(
            list(basis),
            PointChargeIntegral.boys_func,
            compute_two_electron_integrals_os_hgp,
            threshold=1e-10,
        )
        stats = screener.get_statistics()
        # Far-apart shells should have some bounds below threshold
        assert stats["total"] == 0  # No quartets checked yet via is_significant

        # Now run full computation with screening
        result = electron_repulsion_integral_improved(
            basis, notation="chemist", schwarz_threshold=1e-10
        )
        assert np.all(np.isfinite(result))
