"""Improved two-electron integrals using Obara-Saika + Head-Gordon-Pople recursions.

This module implements VRR, ETR, and primitive contraction steps of the
OS+HGP algorithm for two-electron integrals.

Algorithm overview (full pipeline):
1. Start with Boys function F_m(T) for m = 0 to angmom_total
2. VRR: Build [a0|00]^m from [00|00]^m (Eq. 65)        <-- Done (PR 2)
3. ETR: Build [a0|c0]^0 from [a0|00]^m (Eq. 66)        <-- THIS PR
4. Contract primitives                                    <-- THIS PR
5. HRR: Build [ab|cd] from [a0|c0] (Eq. 67)             <-- Future

References:
- Obara, S. & Saika, A. J. Chem. Phys. 1986, 84, 3963.
- Head-Gordon, M. & Pople, J. A. J. Chem. Phys. 1988, 89, 5777.
- Ahlrichs, R. Phys. Chem. Chem. Phys. 2006, 8, 3072.
"""

import functools

import numpy as np

from gbasis.utils import factorial2


@functools.cache
def _get_factorial2_norm(angmom_key):
    """Get cached factorial2 normalization for angular momentum components.

    Parameters
    ----------
    angmom_key : tuple of tuples
        Angular momentum components as a tuple of tuples, e.g.
        ((lx1, ly1, lz1), (lx2, ly2, lz2), ...).

    Returns
    -------
    norm : np.ndarray(n,)
        Normalization factors 1/sqrt(prod((2*l-1)!!)).
    """
    angmom_components = np.array(angmom_key)
    return 1.0 / np.sqrt(np.prod(factorial2(2 * angmom_components - 1), axis=1))


def _optimized_contraction(integrals_etransf, exps, coeffs, angmoms):
    """Optimized primitive contraction using einsum.

    Parameters
    ----------
    integrals_etransf : np.ndarray
        ETR output with shape (c_x, c_y, c_z, a_x, a_y, a_z, K_d, K_b, K_c, K_a).
    exps : array-like of shape (4, K)
        Primitive exponents stacked for all 4 centers (a, b, c, d).
    coeffs : array-like of shape (4, K, M)
        Contraction coefficients stacked for all 4 centers (a, b, c, d).
    angmoms : array-like of shape (4,)
        Angular momenta for all 4 centers (a, b, c, d).

    Returns
    -------
    contracted : np.ndarray
        Contracted integrals with shape (c_x, c_y, c_z, a_x, a_y, a_z, M_a, M_c, M_b, M_d).
    """
    # Vectorized norm computation over all 4 centers
    exps = np.array(exps)  # shape (4, K)
    angmoms = np.array(angmoms)  # shape (4,)
    coeffs = np.array(coeffs)  # shape (4, K, M)
    # shape (4, K)
    norms = ((2 / np.pi) * exps) ** 0.75 * (4 * exps) ** (angmoms[:, np.newaxis] / 2)
    coeffs_norm = coeffs * norms[:, :, np.newaxis]  # shape (4, K, M)
    coeffs_a_norm, coeffs_b_norm, coeffs_c_norm, coeffs_d_norm = coeffs_norm

    # Use einsum with optimization for contraction
    # Input: (c_x, c_y, c_z, a_x, a_y, a_z, K_d, K_b, K_c, K_a)
    # Contract one primitive at a time for memory efficiency
    # K_a contraction
    contracted = np.einsum("...a,aA->...A", integrals_etransf, coeffs_a_norm, optimize=True)
    # K_c contraction (now axis -2 is K_c)
    contracted = np.einsum("...cA,cC->...CA", contracted, coeffs_c_norm, optimize=True)
    # K_b contraction (now axis -3 is K_b)
    contracted = np.einsum("...bCA,bB->...CBA", contracted, coeffs_b_norm, optimize=True)
    # K_d contraction (now axis -4 is K_d)
    contracted = np.einsum("...dCBA,dD->...CBAD", contracted, coeffs_d_norm, optimize=True)

    # Reorder to (c_x, c_y, c_z, a_x, a_y, a_z, M_a, M_c, M_b, M_d)
    # Current: (..., M_c, M_b, M_a, M_d) -> need (..., M_a, M_c, M_b, M_d)
    contracted = np.moveaxis(contracted, -2, -4)

    return contracted


def _vertical_recursion_relation(
    integrals_m,
    m_max,
    rel_coord_a,
    coord_wac,
    harm_mean,
    exps_sum_one,
):
    """Apply Vertical Recursion Relation (VRR) to build angular momentum on center A.

    This implements Eq. 65 from the algorithm notes:
    [a+1,0|00]^m = (P-A)_i [a0|00]^m - (rho/zeta)(Q-P)_i [a0|00]^{m+1}
                   + a_i/(2*zeta) * ([a-1,0|00]^m - (rho/zeta)[a-1,0|00]^{m+1})

    Parameters
    ----------
    integrals_m : np.ndarray
        Array containing [00|00]^m for all m values.
        Shape: (m_max, K_d, K_b, K_c, K_a)
    m_max : int
        Maximum angular momentum order.
    rel_coord_a : np.ndarray(3, K_d, K_b, K_c, K_a)
        P - A coordinates for each primitive combination.
    coord_wac : np.ndarray(3, K_d, K_b, K_c, K_a)
        Q - P coordinates (weighted average centers difference).
    harm_mean : np.ndarray(K_d, K_b, K_c, K_a)
        Harmonic mean: rho = zeta*eta/(zeta+eta).
    exps_sum_one : np.ndarray(K_d, K_b, K_c, K_a)
        Sum of exponents: zeta = alpha_a + alpha_b.

    Returns
    -------
    integrals_vert : np.ndarray
        Integrals with angular momentum built on center A.
        Shape: (m_max, m_max, m_max, m_max, K_d, K_b, K_c, K_a)
        Axes 1, 2, 3 correspond to a_x, a_y, a_z.
    """
    # Precompute coefficients for efficiency (avoid repeated division)
    rho_over_zeta = harm_mean / exps_sum_one
    half_over_zeta = 0.5 / exps_sum_one

    # Precompute products (avoids repeated multiplication in loops)
    roz_wac_x = rho_over_zeta * coord_wac[0]
    roz_wac_y = rho_over_zeta * coord_wac[1]
    roz_wac_z = rho_over_zeta * coord_wac[2]

    # Initialize output array with contiguous memory
    # axis 0: m, axis 1: a_x, axis 2: a_y, axis 3: a_z
    integrals_vert = np.zeros((m_max, m_max, m_max, m_max, *integrals_m.shape[1:]), order="C")

    # Base case: [00|00]^m
    integrals_vert[:, 0, 0, 0, ...] = integrals_m

    # VRR for x-component (a_x)
    if m_max > 1:
        # First step: a_x = 0 -> a_x = 1
        integrals_vert[:-1, 1, 0, 0, ...] = (
            rel_coord_a[0] * integrals_vert[:-1, 0, 0, 0, ...]
            - roz_wac_x * integrals_vert[1:, 0, 0, 0, ...]
        )
        # Higher a_x values (precompute a * half_over_zeta)
        for a in range(1, m_max - 1):
            coeff_a = a * half_over_zeta
            integrals_vert[:-1, a + 1, 0, 0, ...] = (
                rel_coord_a[0] * integrals_vert[:-1, a, 0, 0, ...]
                - roz_wac_x * integrals_vert[1:, a, 0, 0, ...]
                + coeff_a
                * (
                    integrals_vert[:-1, a - 1, 0, 0, ...]
                    - rho_over_zeta * integrals_vert[1:, a - 1, 0, 0, ...]
                )
            )

    # VRR for y-component (a_y)
    if m_max > 1:
        integrals_vert[:-1, :, 1, 0, ...] = (
            rel_coord_a[1] * integrals_vert[:-1, :, 0, 0, ...]
            - roz_wac_y * integrals_vert[1:, :, 0, 0, ...]
        )
        for a in range(1, m_max - 1):
            coeff_a = a * half_over_zeta
            integrals_vert[:-1, :, a + 1, 0, ...] = (
                rel_coord_a[1] * integrals_vert[:-1, :, a, 0, ...]
                - roz_wac_y * integrals_vert[1:, :, a, 0, ...]
                + coeff_a
                * (
                    integrals_vert[:-1, :, a - 1, 0, ...]
                    - rho_over_zeta * integrals_vert[1:, :, a - 1, 0, ...]
                )
            )

    # VRR for z-component (a_z)
    if m_max > 1:
        integrals_vert[:-1, :, :, 1, ...] = (
            rel_coord_a[2] * integrals_vert[:-1, :, :, 0, ...]
            - roz_wac_z * integrals_vert[1:, :, :, 0, ...]
        )
        for a in range(1, m_max - 1):
            coeff_a = a * half_over_zeta
            integrals_vert[:-1, :, :, a + 1, ...] = (
                rel_coord_a[2] * integrals_vert[:-1, :, :, a, ...]
                - roz_wac_z * integrals_vert[1:, :, :, a, ...]
                + coeff_a
                * (
                    integrals_vert[:-1, :, :, a - 1, ...]
                    - rho_over_zeta * integrals_vert[1:, :, :, a - 1, ...]
                )
            )

    return integrals_vert


def _electron_transfer_recursion(
    integrals_vert,
    m_max,
    m_max_c,
    rel_coord_c,
    rel_coord_a,
    exps_sum_one,
    exps_sum_two,
):
    """Apply Electron Transfer Recursion (ETR) to transfer angular momentum to center C.

    This implements Eq. 66 from the algorithm notes:
    [a0|c+1,0]^0 = (Q-C)_i [a0|c0]^0 + (zeta/eta)(P-A)_i [a0|c0]^0
                   - (zeta/eta) [a+1,0|c0]^0
                   + c_i/(2*eta) [a0|c-1,0]^0
                   + a_i/(2*eta) [a-1,0|c0]^0

    Parameters
    ----------
    integrals_vert : np.ndarray
        Output from VRR with angular momentum on A.
        Shape: (m_max, a_x_max, a_y_max, a_z_max, K_d, K_b, K_c, K_a)
    m_max : int
        Maximum m value (angmom_a + angmom_b + angmom_c + angmom_d + 1).
    m_max_c : int
        Maximum c angular momentum (angmom_c + angmom_d + 1).
    rel_coord_c : np.ndarray(3, K_d, K_b, K_c, K_a)
        Q - C coordinates.
    rel_coord_a : np.ndarray(3, K_d, K_b, K_c, K_a)
        P - A coordinates.
    exps_sum_one : np.ndarray
        zeta = alpha_a + alpha_b.
    exps_sum_two : np.ndarray
        eta = alpha_c + alpha_d.

    Returns
    -------
    integrals_etransf : np.ndarray
        Integrals with angular momentum on both A and C.
        Shape: (c_x_max, c_y_max, c_z_max, a_x_max, a_y_max, a_z_max, K_d, K_b, K_c, K_a)
    """
    n_primitives = integrals_vert.shape[4:]
    zeta_over_eta = exps_sum_one / exps_sum_two

    # Precompute coefficients (avoid repeated division in loops)
    half_over_eta = 0.5 / exps_sum_two

    # Precompute combined coordinate terms for each axis
    qc_plus_zoe_pa_x = rel_coord_c[0] + zeta_over_eta * rel_coord_a[0]
    qc_plus_zoe_pa_y = rel_coord_c[1] + zeta_over_eta * rel_coord_a[1]
    qc_plus_zoe_pa_z = rel_coord_c[2] + zeta_over_eta * rel_coord_a[2]

    # Initialize ETR output with contiguous memory
    integrals_etransf = np.zeros(
        (m_max_c, m_max_c, m_max_c, m_max, m_max, m_max, *n_primitives), order="C"
    )

    # Base case: discard m index (take m=0)
    integrals_etransf[0, 0, 0, ...] = integrals_vert[0, ...]

    # Precompute a_indices coefficient array once
    if m_max > 2:
        a_coeff_x = (
            np.arange(1, m_max - 1).reshape(-1, 1, 1, *([1] * len(n_primitives))) * half_over_eta
        )
        a_coeff_y = (
            np.arange(1, m_max - 1).reshape(1, 1, 1, 1, -1, 1, *([1] * len(n_primitives)))
            * half_over_eta
        )
        a_coeff_z = (
            np.arange(1, m_max - 1).reshape(1, 1, 1, 1, 1, -1, *([1] * len(n_primitives)))
            * half_over_eta
        )

    # ETR for c_x
    for c in range(m_max_c - 1):
        c_coeff = c * half_over_eta  # Precompute c/(2*eta)
        if c == 0:
            # First step: c_x = 0 -> c_x = 1
            # For a_x = 0
            integrals_etransf[1, 0, 0, 0, ...] = (
                qc_plus_zoe_pa_x * integrals_etransf[0, 0, 0, 0, ...]
                - zeta_over_eta * integrals_etransf[0, 0, 0, 1, ...]
            )
            # For a_x >= 1
            if m_max > 2:
                integrals_etransf[1, 0, 0, 1:-1, ...] = (
                    qc_plus_zoe_pa_x * integrals_etransf[0, 0, 0, 1:-1, ...]
                    + a_coeff_x * integrals_etransf[0, 0, 0, :-2, ...]
                    - zeta_over_eta * integrals_etransf[0, 0, 0, 2:, ...]
                )
        else:
            # General case: c_x -> c_x + 1
            integrals_etransf[c + 1, 0, 0, 0, ...] = (
                qc_plus_zoe_pa_x * integrals_etransf[c, 0, 0, 0, ...]
                + c_coeff * integrals_etransf[c - 1, 0, 0, 0, ...]
                - zeta_over_eta * integrals_etransf[c, 0, 0, 1, ...]
            )
            if m_max > 2:
                integrals_etransf[c + 1, 0, 0, 1:-1, ...] = (
                    qc_plus_zoe_pa_x * integrals_etransf[c, 0, 0, 1:-1, ...]
                    + a_coeff_x * integrals_etransf[c, 0, 0, :-2, ...]
                    + c_coeff * integrals_etransf[c - 1, 0, 0, 1:-1, ...]
                    - zeta_over_eta * integrals_etransf[c, 0, 0, 2:, ...]
                )

    # ETR for c_y (similar structure, using precomputed coefficients)
    for c in range(m_max_c - 1):
        c_coeff = c * half_over_eta
        if c == 0:
            integrals_etransf[:, 1, 0, :, 0, ...] = (
                qc_plus_zoe_pa_y * integrals_etransf[:, 0, 0, :, 0, ...]
                - zeta_over_eta * integrals_etransf[:, 0, 0, :, 1, ...]
            )
            if m_max > 2:
                integrals_etransf[:, 1, 0, :, 1:-1, ...] = (
                    qc_plus_zoe_pa_y * integrals_etransf[:, 0, 0, :, 1:-1, ...]
                    + a_coeff_y * integrals_etransf[:, 0, 0, :, :-2, ...]
                    - zeta_over_eta * integrals_etransf[:, 0, 0, :, 2:, ...]
                )
        else:
            integrals_etransf[:, c + 1, 0, :, 0, ...] = (
                qc_plus_zoe_pa_y * integrals_etransf[:, c, 0, :, 0, ...]
                + c_coeff * integrals_etransf[:, c - 1, 0, :, 0, ...]
                - zeta_over_eta * integrals_etransf[:, c, 0, :, 1, ...]
            )
            if m_max > 2:
                integrals_etransf[:, c + 1, 0, :, 1:-1, ...] = (
                    qc_plus_zoe_pa_y * integrals_etransf[:, c, 0, :, 1:-1, ...]
                    + a_coeff_y * integrals_etransf[:, c, 0, :, :-2, ...]
                    + c_coeff * integrals_etransf[:, c - 1, 0, :, 1:-1, ...]
                    - zeta_over_eta * integrals_etransf[:, c, 0, :, 2:, ...]
                )

    # ETR for c_z (similar structure, using precomputed coefficients)
    for c in range(m_max_c - 1):
        c_coeff = c * half_over_eta
        if c == 0:
            integrals_etransf[:, :, 1, :, :, 0, ...] = (
                qc_plus_zoe_pa_z * integrals_etransf[:, :, 0, :, :, 0, ...]
                - zeta_over_eta * integrals_etransf[:, :, 0, :, :, 1, ...]
            )
            if m_max > 2:
                integrals_etransf[:, :, 1, :, :, 1:-1, ...] = (
                    qc_plus_zoe_pa_z * integrals_etransf[:, :, 0, :, :, 1:-1, ...]
                    + a_coeff_z * integrals_etransf[:, :, 0, :, :, :-2, ...]
                    - zeta_over_eta * integrals_etransf[:, :, 0, :, :, 2:, ...]
                )
        else:
            integrals_etransf[:, :, c + 1, :, :, 0, ...] = (
                qc_plus_zoe_pa_z * integrals_etransf[:, :, c, :, :, 0, ...]
                + c_coeff * integrals_etransf[:, :, c - 1, :, :, 0, ...]
                - zeta_over_eta * integrals_etransf[:, :, c, :, :, 1, ...]
            )
            if m_max > 2:
                integrals_etransf[:, :, c + 1, :, :, 1:-1, ...] = (
                    qc_plus_zoe_pa_z * integrals_etransf[:, :, c, :, :, 1:-1, ...]
                    + a_coeff_z * integrals_etransf[:, :, c, :, :, :-2, ...]
                    + c_coeff * integrals_etransf[:, :, c - 1, :, :, 1:-1, ...]
                    - zeta_over_eta * integrals_etransf[:, :, c, :, :, 2:, ...]
                )

    return integrals_etransf
