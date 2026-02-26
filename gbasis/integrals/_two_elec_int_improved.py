"""Improved two-electron integrals using Obara-Saika + Head-Gordon-Pople recursions.

This module implements the Vertical Recursion Relation (VRR) for building
angular momentum on center A, as the first step of the OS+HGP algorithm.

Algorithm overview (full pipeline, to be completed in future PRs):
1. Start with Boys function F_m(T) for m = 0 to angmom_total
2. VRR: Build [a0|00]^m from [00|00]^m (Eq. 65)        <-- THIS PR
3. ETR: Build [a0|c0]^0 from [a0|00]^m (Eq. 66)        <-- Future
4. Contract primitives                                    <-- Future
5. HRR: Build [ab|cd] from [a0|c0] (Eq. 67)             <-- Future

References:
- Obara, S. & Saika, A. J. Chem. Phys. 1986, 84, 3963.
- Head-Gordon, M. & Pople, J. A. J. Chem. Phys. 1988, 89, 5777.
- Ahlrichs, R. Phys. Chem. Chem. Phys. 2006, 8, 3072.
"""

import numpy as np


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
