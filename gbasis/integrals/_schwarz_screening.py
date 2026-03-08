"""Integral screening utilities for efficient 2-electron integral computation.

This module implements Schwarz screening and shell-pair screening to skip
negligible integrals, providing speedup for spatially extended systems.

References:
- Häser, M. & Ahlrichs, R. J. Comput. Chem. 1989, 10, 104.
- Gill, P. M. W.; Johnson, B. G.; Pople, J. A. Int. J. Quantum Chem. 1991, 40, 745.
"""

import numpy as np


def compute_schwarz_bound_shell_pair(boys_func, cont_one, cont_two, compute_integral_func):
    """Compute Schwarz bound for a shell pair: sqrt((ab|ab)).

    Parameters
    ----------
    boys_func : callable
        Boys function for integral evaluation.
    cont_one : GeneralizedContractionShell
        First contracted shell.
    cont_two : GeneralizedContractionShell
        Second contracted shell.
    compute_integral_func : callable
        Function to compute (ab|cd) integrals.

    Returns
    -------
    bound : float
        Schwarz bound sqrt(max|(ab|ab)|) for this shell pair.
    """
    # Compute (ab|ab) integral
    integral = compute_integral_func(
        boys_func,
        cont_one.coord,
        cont_one.angmom,
        cont_one.angmom_components_cart,
        cont_one.exps,
        cont_one.coeffs,
        cont_two.coord,
        cont_two.angmom,
        cont_two.angmom_components_cart,
        cont_two.exps,
        cont_two.coeffs,
        cont_one.coord,
        cont_one.angmom,
        cont_one.angmom_components_cart,
        cont_one.exps,
        cont_one.coeffs,
        cont_two.coord,
        cont_two.angmom,
        cont_two.angmom_components_cart,
        cont_two.exps,
        cont_two.coeffs,
    )

    # Return sqrt of maximum absolute value
    return np.sqrt(np.max(np.abs(integral)))


def compute_schwarz_bounds(contractions, boys_func, compute_integral_func):
    """Precompute Schwarz bounds for all shell pairs.

    Parameters
    ----------
    contractions : list of GeneralizedContractionShell
        List of all contracted shells.
    boys_func : callable
        Boys function for integral evaluation.
    compute_integral_func : callable
        Function to compute (ab|cd) integrals.

    Returns
    -------
    bounds : np.ndarray(n_shells, n_shells)
        Schwarz bounds sqrt((ab|ab)) for each shell pair.
    """
    n_shells = len(contractions)
    bounds = np.zeros((n_shells, n_shells))

    for i, cont_i in enumerate(contractions):
        for j in range(i, n_shells):
            cont_j = contractions[j]
            bounds[i, j] = compute_schwarz_bound_shell_pair(
                boys_func, cont_i, cont_j, compute_integral_func
            )
            bounds[j, i] = bounds[i, j]  # Symmetry: (ab|ab) = (ba|ba)

    return bounds


def shell_pair_significant(cont_one, cont_two, threshold=1e-12):
    """Check if a shell pair is significant using primitive screening.

    Uses the Gaussian product theorem: exp(-a*b/(a+b) * |A-B|^2) factor.
    If this factor is below threshold for all primitive pairs, skip.

    Parameters
    ----------
    cont_one : GeneralizedContractionShell
        First contracted shell.
    cont_two : GeneralizedContractionShell
        Second contracted shell.
    threshold : float
        Screening threshold.

    Returns
    -------
    significant : bool
        True if shell pair might contribute significantly.
    """
    # Distance between shell centers
    r_ab_sq = np.sum((cont_one.coord - cont_two.coord) ** 2)

    if r_ab_sq < 1e-10:
        # Same center, always significant
        return True

    # Check if any primitive pair survives screening
    for exp_a in cont_one.exps:
        for exp_b in cont_two.exps:
            # Gaussian decay factor
            decay = np.exp(-exp_a * exp_b / (exp_a + exp_b) * r_ab_sq)
            if decay > threshold:
                return True

    return False


class SchwarzScreener:
    """Class for Schwarz integral screening.

    Precomputes Schwarz bounds and provides efficient screening.

    Attributes
    ----------
    bounds : np.ndarray
        Schwarz bounds for all shell pairs.
    threshold : float
        Screening threshold.
    n_screened : int
        Counter for number of screened shell quartets.
    n_computed : int
        Counter for number of computed shell quartets.
    """

    def __init__(self, contractions, boys_func, compute_integral_func, threshold=1e-12):
        """Initialize Schwarz screener.

        Parameters
        ----------
        contractions : list of GeneralizedContractionShell
            List of all contracted shells.
        boys_func : callable
            Boys function for integral evaluation.
        compute_integral_func : callable
            Function to compute (ab|cd) integrals.
        threshold : float
            Screening threshold (default: 1e-12).
        """
        self.threshold = threshold
        self.n_screened = 0
        self.n_computed = 0

        # Precompute Schwarz bounds
        self.bounds = compute_schwarz_bounds(contractions, boys_func, compute_integral_func)

    def is_significant(self, i, j, k, l_shell):
        """Check if shell quartet (ij|kl) is significant.

        Uses Schwarz inequality: |(ij|kl)| <= sqrt((ij|ij)) * sqrt((kl|kl))

        Parameters
        ----------
        i, j, k, l_shell : int
            Shell indices.

        Returns
        -------
        significant : bool
            True if integral might be significant, False if can be skipped.
        """
        bound = self.bounds[i, j] * self.bounds[k, l_shell]

        if bound < self.threshold:
            self.n_screened += 1
            return False
        else:
            self.n_computed += 1
            return True

    def get_statistics(self):
        """Get screening statistics.

        Returns
        -------
        stats : dict
            Dictionary with screening statistics.
        """
        total = self.n_screened + self.n_computed
        if total == 0:
            percent_screened = 0.0
        else:
            percent_screened = 100.0 * self.n_screened / total

        return {
            "n_screened": self.n_screened,
            "n_computed": self.n_computed,
            "total": total,
            "percent_screened": percent_screened,
            "speedup_factor": total / max(self.n_computed, 1),
        }

    def reset_counters(self):
        """Reset screening counters."""
        self.n_screened = 0
        self.n_computed = 0
