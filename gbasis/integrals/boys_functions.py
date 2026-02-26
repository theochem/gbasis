"""Boys functions for two-electron interaction potentials.

This module implements the Boys function, which is the starting point for
computing two-electron integrals using the Obara-Saika recursion relations.

The Boys function is defined as:

    F_m(T) = integral from 0 to 1 of t^(2m) * exp(-T*t^2) dt

References:
- Helgaker, T.; Jorgensen, P.; Olsen, J. "Molecular Electronic-Structure Theory"
  (eq. 9.8.39 for hyp1f1 representation)
- Ahlrichs, R. "A simple algebraic derivation of the Obara-Saika scheme for
  general two-electron interaction potentials."
  Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
"""

import numpy as np
from scipy.special import hyp1f1


def boys_function_standard(orders, weighted_dist):
    r"""Compute standard Boys function for Coulomb potential (1/r12).

    The Coulombic Boys function is defined as:

    .. math::

        F_m(T) = \int_0^1 t^{2m} e^{-T t^2} dt

    This can be expressed in terms of the Kummer confluent hypergeometric
    function (hyp1f1) as shown in Helgaker (eq. 9.8.39).

    Parameters
    ----------
    orders : np.ndarray
        Differentiation order of the Boys function (m values).
        Shape can be (M,) or (M, 1, 1, ...) for broadcasting.
    weighted_dist : np.ndarray
        Weighted interatomic distances (T values).
        T = rho * |P - Q|^2 where rho = zeta*eta/(zeta+eta).

    Returns
    -------
    boys_eval : np.ndarray
        Boys function values F_m(T) with shape determined by broadcasting
        of orders and weighted_dist.

    Notes
    -----
    For the Coulomb potential g(r) = 1/r:
    - G_0(rho, T) = (2*pi/rho) * F_0(T)

    This is the standard case used for electron repulsion integrals.

    There's some documented instability for hyp1f1, mainly for large values.
    For typical quantum chemistry calculations (m < 20), the values are stable.
    """
    return hyp1f1(orders + 0.5, orders + 1.5, -weighted_dist) / (2 * orders + 1)


def boys_function_all_orders(m_max, weighted_dist):
    r"""Compute Boys function for all orders from 0 to m_max.

    Returns F_m(T) for m = 0, 1, ..., m_max simultaneously using
    scipy's hyp1f1, which is numerically stable for all practical T
    and m values encountered in quantum chemistry.

    Parameters
    ----------
    m_max : int
        Maximum order of the Boys function needed.
    weighted_dist : np.ndarray
        Weighted interatomic distances (T values).

    Returns
    -------
    boys_all : np.ndarray
        Boys function values for all orders from 0 to m_max.
        Shape: (m_max + 1, *weighted_dist.shape)
    """
    # Vectorize hyp1f1 across all orders with broadcasting to avoid
    # per-order Python loops while keeping numerical behaviour identical
    # to the standard expression. hyp1f1 is stable for the practical
    # range m < 20 and T encountered here.

    T = np.asarray(weighted_dist)
    orders = np.arange(m_max + 1, dtype=np.result_type(T, np.float64))

    # Reshape orders to broadcast over T's dimensions: (m_max+1, 1, 1, ...)
    orders_shape = (m_max + 1,) + (1,) * T.ndim
    orders_b = orders.reshape(orders_shape)

    boys_vals = hyp1f1(orders_b + 0.5, orders_b + 1.5, -T) / (2 * orders_b + 1)
    return boys_vals


def boys_function_recursion(m_max, weighted_dist):
    r"""Compute Boys function for all orders using downward recursion.

    Computes F_m(T) for m = 0, 1, ..., m_max using the stable downward
    recursion (Eq. 71 from algorithm notes):

    .. math::

        F_m(T) = \frac{2T \cdot F_{m+1}(T) + e^{-T}}{2m + 1}

    The highest order F_{m_max}(T) is computed via hyp1f1 as the starting
    value, then all lower orders are obtained by recursing downward.
    This is more efficient than calling hyp1f1 for each order separately.

    Parameters
    ----------
    m_max : int
        Maximum order of the Boys function needed.
    weighted_dist : np.ndarray
        Weighted interatomic distances (T values).

    Returns
    -------
    boys_all : np.ndarray
        Boys function values for all orders from 0 to m_max.
        Shape: (m_max + 1, *weighted_dist.shape)
    """
    T = np.asarray(weighted_dist, dtype=np.float64)
    result = np.empty((m_max + 1, *T.shape), dtype=np.float64)

    # Starting value: compute F_{m_max}(T) via hyp1f1
    result[m_max] = hyp1f1(m_max + 0.5, m_max + 1.5, -T) / (2 * m_max + 1)

    # Precompute exp(-T) once
    exp_neg_T = np.exp(-T)

    # Downward recursion: F_m = (2T * F_{m+1} + exp(-T)) / (2m + 1)
    for m in range(m_max - 1, -1, -1):
        result[m] = (2.0 * T * result[m + 1] + exp_neg_T) / (2 * m + 1)

    return result


def boys_function_erf(orders, weighted_dist, rho, omega):
    r"""Compute Boys function for erf-attenuated Coulomb potential.

    For the potential g(r) = erf(omega * r) / r, the modified Boys function is:

    .. math::

        G_m(\rho, T) = \frac{2\pi}{\rho} \left(\frac{\omega^2}{\omega^2 + \rho}\right)^{m+1/2}
                       F_m\left(\frac{\omega^2 T}{\omega^2 + \rho}\right)

    This is used in range-separated DFT for the long-range part of the
    electron-electron interaction.

    Parameters
    ----------
    orders : np.ndarray
        Differentiation orders (m values).
    weighted_dist : np.ndarray
        Weighted interatomic distances (T = rho * |P-Q|^2).
    rho : np.ndarray
        Harmonic mean of exponent sums: rho = zeta*eta/(zeta+eta).
    omega : float
        Range-separation parameter.

    Returns
    -------
    boys_eval : np.ndarray
        Modified Boys function values.

    Notes
    -----
    As omega -> infinity, erf(omega*r)/r -> 1/r, recovering the standard Coulomb.
    As omega -> 0, erf(omega*r)/r -> 0, the interaction vanishes.
    """
    scaling = omega**2 / (omega**2 + rho)
    T_modified = scaling * weighted_dist
    return scaling ** (orders + 0.5) * boys_function_standard(orders, T_modified)


def boys_function_erfc(orders, weighted_dist, rho, omega):
    r"""Compute Boys function for erfc-attenuated Coulomb potential.

    For the potential g(r) = erfc(omega * r) / r, which is the short-range
    complement of the erf-attenuated potential:

    .. math::

        G_m^{erfc} = G_m^{Coulomb} - G_m^{erf}

    This is used in range-separated DFT for the short-range part of the
    electron-electron interaction.

    Parameters
    ----------
    orders : np.ndarray
        Differentiation orders (m values).
    weighted_dist : np.ndarray
        Weighted interatomic distances (T = rho * |P-Q|^2).
    rho : np.ndarray
        Harmonic mean of exponent sums: rho = zeta*eta/(zeta+eta).
    omega : float
        Range-separation parameter.

    Returns
    -------
    boys_eval : np.ndarray
        Modified Boys function values.

    Notes
    -----
    erfc(omega*r)/r = 1/r - erf(omega*r)/r, so:
    G_m^{erfc} = F_m(T) - (omega^2/(omega^2+rho))^(m+0.5) * F_m(omega^2*T/(omega^2+rho))

    As omega -> infinity, erfc(omega*r)/r -> 0 (short-range vanishes).
    As omega -> 0, erfc(omega*r)/r -> 1/r (full Coulomb).
    """
    return boys_function_standard(orders, weighted_dist) - boys_function_erf(
        orders, weighted_dist, rho, omega
    )


def boys_function_mpmath(m_max, weighted_dist, dps=50):
    r"""Compute Boys function using mpmath for high-precision reference values.

    Uses mpmath.hyp1f1 with arbitrary precision arithmetic. Useful as a
    reference for validating other implementations, or for edge cases where
    scipy's hyp1f1 may lose precision (very high m or specific T ranges).

    Parameters
    ----------
    m_max : int
        Maximum order of the Boys function needed.
    weighted_dist : np.ndarray
        Weighted interatomic distances (T values).
    dps : int
        Decimal places of precision for mpmath (default: 50).

    Returns
    -------
    boys_all : np.ndarray
        Boys function values for all orders from 0 to m_max.
        Shape: (m_max + 1, *weighted_dist.shape)

    Raises
    ------
    ImportError
        If mpmath is not installed.

    Notes
    -----
    This function is SLOW compared to hyp1f1 or recursion-based methods.
    It is intended only for validation and reference, not for production use.
    Install mpmath via: pip install mpmath
    """
    try:
        import mpmath
    except ImportError as err:
        raise ImportError(
            "mpmath is required for boys_function_mpmath. Install it with: pip install mpmath"
        ) from err

    T = np.asarray(weighted_dist, dtype=np.float64)
    result = np.empty((m_max + 1, *T.shape), dtype=np.float64)

    old_dps = mpmath.mp.dps
    mpmath.mp.dps = dps
    try:
        for idx in np.ndindex(T.shape):
            t_val = mpmath.mpf(float(T[idx]))
            for m in range(m_max + 1):
                val = mpmath.hyp1f1(m + 0.5, m + 1.5, -t_val) / (2 * m + 1)
                result[(m, *idx)] = float(val)
    finally:
        mpmath.mp.dps = old_dps

    return result


def get_boys_function(potential="coulomb", omega=None):
    """Get the appropriate Boys function for a given potential type.

    Parameters
    ----------
    potential : str
        Type of two-electron potential. Options:
        - "coulomb" or "standard" or "1/r": Standard 1/r12 Coulomb potential
        - "erf" or "erf_coulomb": erf-attenuated (long-range) potential
        - "erfc" or "erfc_coulomb": erfc-attenuated (short-range) potential
    omega : float, optional
        Range-separation parameter for erf/erfc potentials.
        Required when potential is "erf" or "erfc".

    Returns
    -------
    boys_func : callable
        Boys function with signature boys_func(orders, weighted_dist, rho=None).

    Raises
    ------
    ValueError
        If potential type is not recognized.
        If omega is not provided for erf/erfc potentials.

    Examples
    --------
    >>> boys = get_boys_function("coulomb")
    >>> F0 = boys(np.array([0]), np.array([1.0]))

    >>> boys_erf = get_boys_function("erf", omega=0.4)
    >>> F0_erf = boys_erf(np.array([0]), np.array([1.0]), rho=0.5)
    """
    potential = potential.lower()

    if potential in ["coulomb", "standard", "1/r"]:

        def boys_func(orders, weighted_dist, rho=None):
            return boys_function_standard(orders, weighted_dist)

        return boys_func

    elif potential in ["erf", "erf_coulomb"]:
        if omega is None:
            raise ValueError("omega parameter is required for erf-attenuated potential")

        def boys_func(orders, weighted_dist, rho=None):
            if rho is None:
                raise ValueError("rho parameter is required for erf-attenuated potential")
            return boys_function_erf(orders, weighted_dist, rho, omega)

        return boys_func

    elif potential in ["erfc", "erfc_coulomb"]:
        if omega is None:
            raise ValueError("omega parameter is required for erfc-attenuated potential")

        def boys_func(orders, weighted_dist, rho=None):
            if rho is None:
                raise ValueError("rho parameter is required for erfc-attenuated potential")
            return boys_function_erfc(orders, weighted_dist, rho, omega)

        return boys_func

    else:
        raise ValueError(
            f"Unknown potential type: {potential}. " f"Valid options: coulomb, erf, erfc"
        )
