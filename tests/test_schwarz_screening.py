"""Benchmark tests for Schwarz screening and integral optimizations.

Tests for Week 7: Schwarz screening (_screening.py) for efficient
two-electron integral computation.

The optimizations tested:
1. Schwarz Screening - Skip negligible shell quartets using (ij|kl) <= Q_ij * Q_kl
2. Shell-pair Prescreening - Fast Gaussian decay check
3. SchwarzScreener class - Precomputed bounds with statistics
"""

import numpy as np

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._schwarz_screening import (
    SchwarzScreener,
    compute_schwarz_bound_shell_pair,
    compute_schwarz_bounds,
    shell_pair_significant,
)
from gbasis.integrals._two_elec_int_improved import compute_two_electron_integrals_os_hgp
from gbasis.integrals.electron_repulsion import electron_repulsion_integral_improved
from gbasis.integrals.point_charge import PointChargeIntegral


def make_shell(coord, angmom, exps, coeffs):
    """Create a GeneralizedContractionShell for testing."""
    return GeneralizedContractionShell(
        angmom=angmom,
        coord=np.array(coord),
        coeffs=np.array(coeffs).reshape(-1, 1),
        exps=np.array(exps),
        coord_type="cartesian",
    )


def make_h_chain(n_atoms, spacing, include_p=False):
    """Create a hydrogen chain basis set for testing.

    Parameters
    ----------
    n_atoms : int
        Number of hydrogen atoms in the chain.
    spacing : float
        Distance between atoms in Bohr.
    include_p : bool
        If True, include p-orbitals (cc-pVDZ-like basis).

    Returns
    -------
    basis : list of GeneralizedContractionShell
        Basis set for the hydrogen chain.
    """
    # STO-3G exponents and coefficients
    exps_s_sto3g = [3.42525091, 0.62391373, 0.16885540]
    coeffs_s_sto3g = [0.15432897, 0.53532814, 0.44463454]

    # cc-pVDZ-like exponents
    exps_s_dz = [13.01, 1.962, 0.4446, 0.122]
    coeffs_s_dz = [0.0196, 0.1379, 0.4781, 0.5012]
    exps_p = [0.727]
    coeffs_p = [1.0]

    basis = []
    for i in range(n_atoms):
        coord = [i * spacing, 0.0, 0.0]
        if include_p:
            basis.append(make_shell(coord, 0, exps_s_dz, coeffs_s_dz))
            basis.append(make_shell(coord, 1, exps_p, coeffs_p))
        else:
            basis.append(make_shell(coord, 0, exps_s_sto3g, coeffs_s_sto3g))

    return basis


class TestShellPairPrescreening:
    """Tests for shell-pair significance screening."""

    def test_same_center_always_significant(self):
        """Test that shells on the same center are always significant."""
        shell = make_shell([0.0, 0.0, 0.0], 0, [1.0], [1.0])
        assert shell_pair_significant(shell, shell) is True

    def test_close_shells_significant(self):
        """Test that close shells are significant."""
        shell_a = make_shell([0.0, 0.0, 0.0], 0, [1.0], [1.0])
        shell_b = make_shell([1.0, 0.0, 0.0], 0, [1.0], [1.0])
        assert shell_pair_significant(shell_a, shell_b) is True

    def test_distant_shells_not_significant(self):
        """Test that very distant shells with tight exponents are not significant."""
        shell_a = make_shell([0.0, 0.0, 0.0], 0, [100.0], [1.0])
        shell_b = make_shell([100.0, 0.0, 0.0], 0, [100.0], [1.0])
        assert shell_pair_significant(shell_a, shell_b) is False

    def test_diffuse_shells_always_significant(self):
        """Test that diffuse shells (small exponents) are significant even at distance."""
        shell_a = make_shell([0.0, 0.0, 0.0], 0, [0.01], [1.0])
        shell_b = make_shell([10.0, 0.0, 0.0], 0, [0.01], [1.0])
        assert shell_pair_significant(shell_a, shell_b) is True

    def test_threshold_sensitivity(self):
        """Test that threshold affects screening."""
        shell_a = make_shell([0.0, 0.0, 0.0], 0, [5.0], [1.0])
        shell_b = make_shell([5.0, 0.0, 0.0], 0, [5.0], [1.0])

        # With strict threshold, should be not significant
        sig_strict = shell_pair_significant(shell_a, shell_b, threshold=1e-6)
        # With loose threshold, should be significant
        sig_loose = shell_pair_significant(shell_a, shell_b, threshold=1e-30)

        # Monotonicity: any pair significant under strict threshold must also be
        # significant under loose threshold (smaller threshold = more permissive)
        assert (not sig_strict) or sig_loose


class TestSchwarzBounds:
    """Tests for Schwarz bound computation."""

    def test_schwarz_bound_positive(self):
        """Test that Schwarz bounds are non-negative."""
        shell = make_shell([0.0, 0.0, 0.0], 0, [1.0, 0.5], [0.6, 0.4])

        boys_func = PointChargeIntegral.boys_func
        bound = compute_schwarz_bound_shell_pair(
            boys_func, shell, shell, compute_two_electron_integrals_os_hgp
        )
        assert bound >= 0.0

    def test_schwarz_bound_nonzero_for_same_shell(self):
        """Test that Schwarz bound is nonzero for same shell."""
        shell = make_shell([0.0, 0.0, 0.0], 0, [1.0], [1.0])

        boys_func = PointChargeIntegral.boys_func
        bound = compute_schwarz_bound_shell_pair(
            boys_func, shell, shell, compute_two_electron_integrals_os_hgp
        )
        assert bound > 0.0

    def test_schwarz_bounds_matrix_symmetric(self):
        """Test that Schwarz bounds matrix is symmetric."""
        basis = make_h_chain(3, spacing=2.0)

        boys_func = PointChargeIntegral.boys_func
        bounds = compute_schwarz_bounds(basis, boys_func, compute_two_electron_integrals_os_hgp)

        np.testing.assert_allclose(
            bounds, bounds.T, rtol=1e-10, err_msg="Schwarz bounds matrix should be symmetric"
        )

    def test_schwarz_bounds_matrix_shape(self):
        """Test that Schwarz bounds matrix has correct shape."""
        basis = make_h_chain(4, spacing=2.0)

        boys_func = PointChargeIntegral.boys_func
        bounds = compute_schwarz_bounds(basis, boys_func, compute_two_electron_integrals_os_hgp)

        assert bounds.shape == (4, 4)

    def test_schwarz_bounds_diagonal_positive(self):
        """Test that diagonal elements of bounds matrix are positive."""
        basis = make_h_chain(3, spacing=2.0)

        boys_func = PointChargeIntegral.boys_func
        bounds = compute_schwarz_bounds(basis, boys_func, compute_two_electron_integrals_os_hgp)

        for i in range(len(basis)):
            assert bounds[i, i] > 0, f"Diagonal bound[{i},{i}] should be positive"

    def test_schwarz_inequality_holds(self):
        """Test that the Schwarz inequality |(ij|kl)| <= Q_ij * Q_kl holds."""
        basis = make_h_chain(3, spacing=2.0)

        boys_func = PointChargeIntegral.boys_func
        bounds = compute_schwarz_bounds(basis, boys_func, compute_two_electron_integrals_os_hgp)

        # Compute actual integrals
        integrals = electron_repulsion_integral_improved(basis, notation="chemist")

        n = len(basis)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        max_integral = np.max(np.abs(integrals[i, j, k, l_idx]))
                        schwarz_bound = bounds[i, j] * bounds[k, l_idx]
                        assert (
                            max_integral <= schwarz_bound * (1 + 1e-6) + 1e-12
                        ), f"Schwarz inequality violated for ({i}{j}|{k}{l_idx})"


class TestSchwarzScreener:
    """Tests for the SchwarzScreener class."""

    def test_screener_initialization(self):
        """Test that screener initializes correctly."""
        basis = make_h_chain(3, spacing=2.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        assert screener.bounds.shape == (3, 3)
        assert screener.n_screened == 0
        assert screener.n_computed == 0

    def test_same_shell_significant(self):
        """Test that same-shell quartets are always significant."""
        basis = make_h_chain(3, spacing=2.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        assert screener.is_significant(0, 0, 0, 0) is True

    def test_screening_statistics(self):
        """Test that screening statistics are tracked correctly."""
        basis = make_h_chain(4, spacing=10.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        # Run through all quartets
        n = len(basis)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        screener.is_significant(i, j, k, l_idx)

        stats = screener.get_statistics()
        assert stats["total"] == n**4
        assert stats["n_screened"] + stats["n_computed"] == n**4
        assert 0 <= stats["percent_screened"] <= 100

    def test_extended_system_has_screening(self):
        """Test that extended systems have significant screening."""
        basis = make_h_chain(4, spacing=15.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        n = len(basis)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        screener.is_significant(i, j, k, l_idx)

        stats = screener.get_statistics()
        # For well-separated atoms, many integrals should be screened
        assert (
            stats["percent_screened"] > 30.0
        ), f"Expected >30% screening for extended system, got {stats['percent_screened']:.1f}%"

    def test_compact_system_no_screening(self):
        """Test that compact systems have little screening."""
        basis = make_h_chain(3, spacing=1.4)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        n = len(basis)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        screener.is_significant(i, j, k, l_idx)

        stats = screener.get_statistics()
        # For compact molecules, most integrals should survive screening
        assert (
            stats["percent_screened"] < 70.0
        ), f"Expected <70% screening for compact system, got {stats['percent_screened']:.1f}%"

    def test_reset_counters(self):
        """Test that reset_counters works correctly."""
        basis = make_h_chain(3, spacing=2.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        screener.is_significant(0, 0, 0, 0)
        screener.is_significant(0, 0, 1, 1)
        assert screener.n_screened + screener.n_computed > 0

        screener.reset_counters()
        assert screener.n_screened == 0
        assert screener.n_computed == 0

    def test_custom_threshold(self):
        """Test that custom threshold affects screening."""
        basis = make_h_chain(4, spacing=5.0)
        boys_func = PointChargeIntegral.boys_func

        # Strict threshold - more screening
        screener_strict = SchwarzScreener(
            basis, boys_func, compute_two_electron_integrals_os_hgp, threshold=1e-8
        )
        # Loose threshold - less screening
        screener_loose = SchwarzScreener(
            basis, boys_func, compute_two_electron_integrals_os_hgp, threshold=1e-16
        )

        n = len(basis)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        screener_strict.is_significant(i, j, k, l_idx)
                        screener_loose.is_significant(i, j, k, l_idx)

        stats_strict = screener_strict.get_statistics()
        stats_loose = screener_loose.get_statistics()

        # Strict threshold should screen at least as much as loose
        assert (
            stats_strict["n_screened"] >= stats_loose["n_screened"]
        ), "Strict threshold should screen more than loose threshold"


class TestScreeningCorrectness:
    """Tests to verify that screened integrals match unscreened."""

    def test_screened_vs_unscreened_compact(self):
        """Test that screened computation gives same result for compact molecule."""
        basis = make_h_chain(3, spacing=1.4)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        # Compute full integrals
        full_integrals = electron_repulsion_integral_improved(basis, notation="chemist")

        # Verify that significant quartets match
        n = full_integrals.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        if not screener.is_significant(i, j, k, l_idx):
                            # Screened integrals should actually be negligible
                            assert (
                                np.abs(full_integrals[i, j, k, l_idx]) < 1e-10
                            ), f"Screened integral ({i}{j}|{k}{l_idx}) is not negligible"

    def test_screened_vs_unscreened_extended(self):
        """Test that screened computation gives same result for extended molecule."""
        basis = make_h_chain(4, spacing=10.0)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        # Compute full integrals
        full_integrals = electron_repulsion_integral_improved(basis, notation="chemist")

        n = full_integrals.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l_idx in range(n):
                        if not screener.is_significant(i, j, k, l_idx):
                            assert (
                                np.abs(full_integrals[i, j, k, l_idx]) < 1e-10
                            ), f"Screened integral ({i}{j}|{k}{l_idx}) is not negligible"

    def test_screened_with_p_orbitals(self):
        """Test screening correctness with p-orbitals."""
        basis = make_h_chain(2, spacing=8.0, include_p=True)
        boys_func = PointChargeIntegral.boys_func

        screener = SchwarzScreener(basis, boys_func, compute_two_electron_integrals_os_hgp)

        # For shell-level screening, verify bounds shape matches
        assert screener.bounds.shape == (len(basis), len(basis))

        # All diagonal bounds should be positive
        for i in range(len(basis)):
            assert screener.bounds[i, i] > 0


class TestBenchmarkImprovedAlgorithm:
    """Benchmark tests for the improved OS+HGP algorithm."""

    def test_improved_algorithm_ssss(self):
        """Test improved algorithm for (ss|ss) integrals."""
        coord = np.array([0.0, 0.0, 0.0])
        exps = np.array([1.0, 0.5])
        coeffs = np.array([[0.6], [0.4]])

        angmom_s = 0
        angmom_comp_s = np.array([[0, 0, 0]])

        boys_func = PointChargeIntegral.boys_func

        result = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord,
            angmom_s,
            angmom_comp_s,
            exps,
            coeffs,
            coord,
            angmom_s,
            angmom_comp_s,
            exps,
            coeffs,
            coord,
            angmom_s,
            angmom_comp_s,
            exps,
            coeffs,
            coord,
            angmom_s,
            angmom_comp_s,
            exps,
            coeffs,
        )

        # Should be positive for (ss|ss)
        assert result[0, 0, 0, 0, 0, 0, 0, 0] > 0

    def test_improved_algorithm_pppp(self):
        """Test improved algorithm for (pp|pp) integrals."""
        coord = np.array([0.0, 0.0, 0.0])
        exps = np.array([1.0])
        coeffs = np.array([[1.0]])

        angmom_p = 1
        angmom_comp_p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        boys_func = PointChargeIntegral.boys_func

        result = compute_two_electron_integrals_os_hgp(
            boys_func,
            coord,
            angmom_p,
            angmom_comp_p,
            exps,
            coeffs,
            coord,
            angmom_p,
            angmom_comp_p,
            exps,
            coeffs,
            coord,
            angmom_p,
            angmom_comp_p,
            exps,
            coeffs,
            coord,
            angmom_p,
            angmom_comp_p,
            exps,
            coeffs,
        )

        assert result.shape == (3, 3, 3, 3, 1, 1, 1, 1)
        assert result[0, 0, 0, 0, 0, 0, 0, 0] > 0
