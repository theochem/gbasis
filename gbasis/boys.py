"""Boys function for 2-electron integral evaluation.

The Boys function F_m(x) = integral_0^1 t^(2m) * exp(-x * t^2) dt
is the key auxiliary function in the Obara-Saika and Head-Gordon-Pople
recurrence relations for 2-electron integrals.
"""

import numpy as np
from scipy.special import hyp1f1, factorial2


def boys_function(m, x):
    """Evaluate the Boys function F_m(x).

    The Boys function is defined as:
        F_m(x) = integral_0^1 t^(2m) * exp(-x * t^2) dt

    For small x, it is evaluated via the confluent hypergeometric function.
    For large x, the asymptotic formula is used.

    Parameters
    ----------
    m : int
        Order of the Boys function (non-negative integer).
    x : float or np.ndarray
        Argument of the Boys function (non-negative).

    Returns
    -------
    float or np.ndarray
        Value of F_m(x).

    Notes
    -----
    This function is central to the Obara-Saika (OS) recurrence relation
    for evaluating 2-electron repulsion integrals [1]_.

    References
    ----------
    .. [1] Obara, S.; Saika, A. J. Chem. Phys. 1986, 84, 3963.
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.empty_like(x)
    large = x > 25.0  # threshold for asymptotic expansion

    # Asymptotic formula for large x: F_m(x) ≈ (2m-1)!! / 2^(m+1) * sqrt(pi/x^(2m+1))
    if np.any(large):
        result[large] = (
            np.math.factorial2(2 * m - 1)
            / 2 ** (m + 1)
            * np.sqrt(np.pi / x[large] ** (2 * m + 1))
        )

    # For small x: use F_m(x) = e^(-x) * sum or hypergeometric form
    # F_m(x) = (1/(2m+1)) * 1F1(m+1/2; m+3/2; -x)
    if np.any(~large):
        result[~large] = (
            1.0 / (2 * m + 1) * hyp1f1(m + 0.5, m + 1.5, -x[~large])
        )

    return result[0] if scalar else result