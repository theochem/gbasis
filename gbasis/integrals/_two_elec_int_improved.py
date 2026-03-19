"""Improved two-electron integrals using Obara-Saika + Head-Gordon-Pople recursions.

This module implements the complete OS+HGP algorithm for two-electron integrals.

Algorithm overview (full pipeline):
1. Start with Boys function F_m(T) for m = 0 to angmom_total    (PR 1)
2. VRR: Build [a0|00]^m from [00|00]^m (Eq. 65)                 (PR 2)
3. ETR: Build [a0|c0]^0 from [a0|00]^m (Eq. 66)                 (PR 3)
4. Contract primitives                                            (PR 3)
5. HRR: Build [ab|cd] from [a0|c0] (Eq. 67)                     (PR 4)

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
    # Compute norms per center (supports different K per center)
    norms = [((2 / np.pi) * e) ** 0.75 * (4 * e) ** (ang / 2) for e, ang in zip(exps, angmoms)]
    coeffs_norm = [c * n[:, np.newaxis] for c, n in zip(coeffs, norms)]
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


def _horizontal_recursion_relation(
    integrals_cont,
    angmom_a,
    angmom_b,
    angmom_c,
    angmom_d,
    angmom_components_a,
    angmom_components_b,
    angmom_components_c,
    angmom_components_d,
    rel_dist_one,
    rel_dist_two,
):
    """Apply Horizontal Recursion Relation (HRR) to distribute angular momentum to B and D.

    This implements Eq. 67 from the algorithm notes (Head-Gordon-Pople scheme):
    [ab|cd] = [a+1,0|cd] + (A-B)_i * [a,0|cd]

    In the HGP scheme, HRR is applied AFTER contraction, which is more efficient
    because contracted integrals are smaller than primitive integrals.

    Parameters
    ----------
    integrals_cont : np.ndarray
        Contracted integrals from ETR+contraction step.
        Shape: (c_x, c_y, c_z, a_x, a_y, a_z, M_a, M_c, M_b, M_d)
    angmom_a/b/c/d : int
        Angular momenta for each center.
    angmom_components_a/b/c/d : np.ndarray(L, 3)
        Angular momentum components for each center.
    rel_dist_one : np.ndarray(3,)
        A - B distance vector.
    rel_dist_two : np.ndarray(3,)
        C - D distance vector.

    Returns
    -------
    integrals : np.ndarray(L_a, L_b, L_c, L_d, M_a, M_b, M_c, M_d)
        Final integrals in Chemists' notation.
    """
    m_max_a = angmom_a + angmom_b + 1
    m_max_c = angmom_c + angmom_d + 1
    n_cont = integrals_cont.shape[6:]  # (M_a, M_c, M_b, M_d)

    # ---- HRR for center D (x and y components) ----
    integrals_horiz_d = np.zeros(
        (angmom_d + 1, angmom_d + 1, m_max_c, m_max_c, m_max_c, m_max_a, m_max_a, m_max_a, *n_cont)
    )
    integrals_horiz_d[0, 0, :, :, :, :, :, :, ...] = integrals_cont[
        :, :, :, :m_max_a, :m_max_a, :m_max_a, ...
    ]

    # HRR x-component of d
    for d in range(angmom_d):
        integrals_horiz_d[d + 1, 0, :-1, :, :, :, :, :, ...] = (
            integrals_horiz_d[d, 0, 1:, :, :, :, :, :, ...]
            + rel_dist_two[0] * integrals_horiz_d[d, 0, :-1, :, :, :, :, :, ...]
        )
    # HRR y-component of d
    for d in range(angmom_d):
        integrals_horiz_d[:, d + 1, :, :-1, :, :, :, :, ...] = (
            integrals_horiz_d[:, d, :, 1:, :, :, :, :, ...]
            + rel_dist_two[1] * integrals_horiz_d[:, d, :, :-1, :, :, :, :, ...]
        )

    # Select angular momentum components (x, y) for c and d, then recurse z
    angmoms_c_x, angmoms_c_y, angmoms_c_z = angmom_components_c.T
    angmoms_d_x, angmoms_d_y, angmoms_d_z = angmom_components_d.T

    integrals_horiz_d2 = np.zeros(
        (
            angmom_d + 1,
            m_max_c,
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            m_max_a,
            m_max_a,
            m_max_a,
            *n_cont,
        )
    )
    integrals_horiz_d2[0] = np.transpose(
        integrals_horiz_d[
            angmoms_d_x.reshape(-1, 1),
            angmoms_d_y.reshape(-1, 1),
            angmoms_c_x.reshape(1, -1),
            angmoms_c_y.reshape(1, -1),
            :,
            :,
            :,
            :,
            :,
            :,
            :,
            :,
        ],
        (2, 0, 1, 3, 4, 5, 6, 7, 8, 9),
    )

    # HRR z-component of d
    for d in range(angmom_d):
        integrals_horiz_d2[d + 1, :-1, :, :, :, :, :, ...] = (
            integrals_horiz_d2[d, 1:, :, :, :, :, :, ...]
            + rel_dist_two[2] * integrals_horiz_d2[d, :-1, :, :, :, :, :, ...]
        )

    # Select z components for c and d
    integrals_horiz_d2 = integrals_horiz_d2[
        angmoms_d_z.reshape(-1, 1),
        angmoms_c_z.reshape(1, -1),
        np.arange(angmoms_d_z.size).reshape(-1, 1),
        np.arange(angmoms_c_z.size).reshape(1, -1),
        :,
        :,
        :,
        :,
        :,
        :,
        :,
    ]

    # ---- HRR for center B (x and y components) ----
    angmoms_a_x, angmoms_a_y, angmoms_a_z = angmom_components_a.T
    angmoms_b_x, angmoms_b_y, angmoms_b_z = angmom_components_b.T

    integrals_horiz_b = np.zeros(
        (
            angmom_b + 1,
            angmom_b + 1,
            m_max_a,
            m_max_a,
            m_max_a,
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            *n_cont,
        )
    )
    integrals_horiz_b[0, 0, :, :, :, :, :, ...] = np.transpose(
        integrals_horiz_d2[:, :, :, :, :, :, :, :, :], (2, 3, 4, 0, 1, 5, 6, 7, 8)
    )

    # HRR x-component of b
    for b in range(angmom_b):
        integrals_horiz_b[b + 1, 0, :-1, :, :, :, :, ...] = (
            integrals_horiz_b[b, 0, 1:, :, :, :, :, ...]
            + rel_dist_one[0] * integrals_horiz_b[b, 0, :-1, :, :, :, :, ...]
        )
    # HRR y-component of b
    for b in range(angmom_b):
        integrals_horiz_b[:, b + 1, :, :-1, :, :, :, ...] = (
            integrals_horiz_b[:, b, :, 1:, :, :, :, ...]
            + rel_dist_one[1] * integrals_horiz_b[:, b, :, :-1, :, :, :, ...]
        )

    # Select angular momentum components (x, y) for a and b, then recurse z
    integrals_horiz_b2 = np.zeros(
        (
            angmom_b + 1,
            m_max_a,
            angmom_components_b.shape[0],
            angmom_components_a.shape[0],
            angmom_components_d.shape[0],
            angmom_components_c.shape[0],
            *n_cont,
        )
    )
    integrals_horiz_b2[0] = np.transpose(
        integrals_horiz_b[
            angmoms_b_x.reshape(-1, 1),
            angmoms_b_y.reshape(-1, 1),
            angmoms_a_x.reshape(1, -1),
            angmoms_a_y.reshape(1, -1),
            :,
            :,
            :,
            :,
            :,
            :,
            :,
        ],
        (2, 0, 1, 3, 4, 5, 6, 7, 8),
    )

    # HRR z-component of b
    for b in range(angmom_b):
        integrals_horiz_b2[b + 1, :-1, :, :, :, :, ...] = (
            integrals_horiz_b2[b, 1:, :, :, :, :, ...]
            + rel_dist_one[2] * integrals_horiz_b2[b, :-1, :, :, :, :, ...]
        )

    # Select z components for a and b
    integrals_horiz_b2 = integrals_horiz_b2[
        angmoms_b_z.reshape(-1, 1),
        angmoms_a_z.reshape(1, -1),
        np.arange(angmoms_b_z.size).reshape(-1, 1),
        np.arange(angmoms_a_z.size).reshape(1, -1),
        :,
        :,
        :,
        :,
        :,
        :,
    ]

    # Rearrange to final order: (L_a, L_b, L_c, L_d, M_a, M_b, M_c, M_d)
    # Current: (L_b, L_a, L_d, L_c, M_a, M_c, M_b, M_d)
    integrals = np.transpose(integrals_horiz_b2, (1, 0, 3, 2, 4, 6, 5, 7))

    # Apply factorial2 normalization for angular momentum components
    norm_a = _get_factorial2_norm(tuple(map(tuple, angmom_components_a))).reshape(
        -1, 1, 1, 1, 1, 1, 1, 1
    )
    norm_b = _get_factorial2_norm(tuple(map(tuple, angmom_components_b))).reshape(
        1, -1, 1, 1, 1, 1, 1, 1
    )
    norm_c = _get_factorial2_norm(tuple(map(tuple, angmom_components_c))).reshape(
        1, 1, -1, 1, 1, 1, 1, 1
    )
    norm_d = _get_factorial2_norm(tuple(map(tuple, angmom_components_d))).reshape(
        1, 1, 1, -1, 1, 1, 1, 1
    )

    integrals = integrals * norm_a * norm_b * norm_c * norm_d

    return integrals


def compute_two_electron_integrals_os_hgp(
    boys_func,
    coord_a,
    angmom_a,
    angmom_components_a,
    exps_a,
    coeffs_a,
    coord_b,
    angmom_b,
    angmom_components_b,
    exps_b,
    coeffs_b,
    coord_c,
    angmom_c,
    angmom_components_c,
    exps_c,
    coeffs_c,
    coord_d,
    angmom_d,
    angmom_components_d,
    exps_d,
    coeffs_d,
    primitive_threshold=0.0,
):
    r"""Compute two-electron integrals using OS+HGP algorithm.

    This is the main entry point that combines all steps:
    1. Boys function initialization (base case [00|00]^m)
    2. VRR: Build angular momentum on center A  (Eq. 65)
    3. ETR: Transfer angular momentum to center C (Eq. 66)
    4. Contract primitives (einsum-based)
    5. HRR: Distribute to centers B and D (Eq. 67, done LAST per HGP)

    Parameters
    ----------
    boys_func : callable
        Boys function with signature boys_func(orders, weighted_dist).
    coord_a/b/c/d : np.ndarray(3,)
        Centers of each contraction.
    angmom_a/b/c/d : int
        Angular momentum of each contraction.
    angmom_components_a/b/c/d : np.ndarray(L, 3)
        Angular momentum components for each contraction.
    exps_a/b/c/d : np.ndarray(K,)
        Primitive exponents.
    coeffs_a/b/c/d : np.ndarray(K, M)
        Contraction coefficients.
    primitive_threshold : float, optional
        Screening threshold for primitive quartets (default: 0.0, no screening).
        Primitive quartets with |prefactor| < threshold are zeroed out per Eq. 64.

    Returns
    -------
    integrals : np.ndarray(L_a, L_b, L_c, L_d, M_a, M_b, M_c, M_d)
        Two-electron integrals in Chemists' notation.
    """
    m_max = angmom_a + angmom_b + angmom_c + angmom_d + 1
    m_max_c = angmom_c + angmom_d + 1

    # --- Pre-compute primitive quantities ---
    # Reshape exponents for broadcasting: (K_d, K_b, K_c, K_a)
    coord_a_4d = coord_a[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_b_4d = coord_b[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_c_4d = coord_c[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    coord_d_4d = coord_d[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    exps_a_4d = exps_a[np.newaxis, np.newaxis, np.newaxis, :]
    exps_b_4d = exps_b[np.newaxis, :, np.newaxis, np.newaxis]
    exps_c_4d = exps_c[np.newaxis, np.newaxis, :, np.newaxis]
    exps_d_4d = exps_d[:, np.newaxis, np.newaxis, np.newaxis]

    # Sum of exponents
    exps_sum_one = exps_a_4d + exps_b_4d  # zeta = alpha_a + alpha_b
    exps_sum_two = exps_c_4d + exps_d_4d  # eta  = alpha_c + alpha_d
    exps_sum = exps_sum_one + exps_sum_two

    # Harmonic means
    harm_mean_one = (exps_a_4d * exps_b_4d) / exps_sum_one
    harm_mean_two = (exps_c_4d * exps_d_4d) / exps_sum_two
    harm_mean = exps_sum_one * exps_sum_two / exps_sum  # rho

    # Weighted average centers
    coord_wac_one = (exps_a_4d * coord_a_4d + exps_b_4d * coord_b_4d) / exps_sum_one  # P
    coord_wac_two = (exps_c_4d * coord_c_4d + exps_d_4d * coord_d_4d) / exps_sum_two  # Q
    coord_wac = coord_wac_one - coord_wac_two  # P - Q

    # Relative coordinates
    rel_dist_one = coord_a - coord_b  # A - B (for HRR)
    rel_dist_two = coord_c - coord_d  # C - D (for HRR)
    rel_coord_a = coord_wac_one - coord_a_4d  # P - A (for VRR)
    rel_coord_c = coord_wac_two - coord_c_4d  # Q - C (for ETR)

    # --- Step 1: Boys function initialization ---
    weighted_dist = harm_mean * np.sum(coord_wac**2, axis=0)
    prefactor = (
        (2 * np.pi**2.5)
        / (exps_sum_one * exps_sum_two * exps_sum**0.5)
        * np.exp(-harm_mean_one * np.sum((coord_a_4d - coord_b_4d) ** 2, axis=0))
        * np.exp(-harm_mean_two * np.sum((coord_c_4d - coord_d_4d) ** 2, axis=0))
    )

    # --- Primitive-level screening (Eq. 64) ---
    if primitive_threshold > 0:
        prefactor = np.where(np.abs(prefactor) >= primitive_threshold, prefactor, 0.0)

    orders = np.arange(m_max)[:, None, None, None, None]
    integrals_m = boys_func(orders, weighted_dist[None, :, :, :, :], rho=harm_mean) * prefactor

    # --- Step 2: VRR ---
    integrals_vert = _vertical_recursion_relation(
        integrals_m, m_max, rel_coord_a, coord_wac, harm_mean, exps_sum_one
    )

    # --- Step 3: ETR ---
    integrals_etransf = _electron_transfer_recursion(
        integrals_vert, m_max, m_max_c, rel_coord_c, rel_coord_a, exps_sum_one, exps_sum_two
    )

    # --- Step 4: Contract primitives ---
    integrals_cont = _optimized_contraction(
        integrals_etransf,
        (exps_a, exps_b, exps_c, exps_d),
        (coeffs_a, coeffs_b, coeffs_c, coeffs_d),
        (angmom_a, angmom_b, angmom_c, angmom_d),
    )

    # --- Step 5: HRR (done LAST per HGP scheme) ---
    integrals = _horizontal_recursion_relation(
        integrals_cont,
        angmom_a,
        angmom_b,
        angmom_c,
        angmom_d,
        angmom_components_a,
        angmom_components_b,
        angmom_components_c,
        angmom_components_d,
        rel_dist_one,
        rel_dist_two,
    )

    return integrals
