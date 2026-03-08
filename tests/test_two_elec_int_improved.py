"""Test gbasis.integrals._two_elec_int_improved module.

Week 2: Tests for VRR (Vertical Recursion Relation).
"""

import numpy as np
import pytest

from gbasis.integrals._two_elec_int_improved import (
    _vertical_recursion_relation,
)


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
        integrals_m = np.array([[[[[1.0]]]],
                                [[[[0.5]]]]])

        PA_x, PA_y, PA_z = 0.3, 0.0, 0.0
        PQ_x, PQ_y, PQ_z = 0.2, 0.0, 0.0
        rho = 0.6
        zeta = 1.5

        rel_coord_a = np.array([[[[[PA_x]]]],
                                [[[[PA_y]]]],
                                [[[[PA_z]]]]])
        coord_wac = np.array([[[[[PQ_x]]]],
                              [[[[PQ_y]]]],
                              [[[[PQ_z]]]]])
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
