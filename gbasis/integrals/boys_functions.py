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


def get_boys_function(potential="coulomb", omega=None):
    """Get the appropriate Boys function for a given potential type.

    Parameters
    ----------
    potential : str
        Type of two-electron potential. Options:
        - "coulomb" or "standard" or "1/r": Standard 1/r12 Coulomb potential
    omega : float, optional
        Range-separation or damping parameter. Reserved for future use
        with screened and damped potentials.

    Returns
    -------
    boys_func : callable
        Boys function with signature boys_func(orders, weighted_dist, rho=None).

    Raises
    ------
    ValueError
        If potential type is not recognized.

    Examples
    --------
    >>> boys = get_boys_function("coulomb")
    >>> F0 = boys(np.array([0]), np.array([1.0]))
    """
    potential = potential.lower()

    if potential in ["coulomb", "standard", "1/r"]:

        def boys_func(orders, weighted_dist, rho=None):
            return boys_function_standard(orders, weighted_dist)

        return boys_func

    else:
        raise ValueError(f"Unknown potential type: {potential}. " f"Valid options: coulomb")
