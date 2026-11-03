"""1 and 2-index screening functions"""

import numpy as np
from scipy.special import lambertw
from gbasis.utils import factorial2


def is_two_index_overlap_screened(contractions_one, contractions_two, tol_screen):
    r"""Return True if the integral should be screened.

    .. math::
           d_{A_s;B_t} > \sqrt{-\frac{\alpha_{ min(\alpha_{s,A})} +
           \alpha_{ min(\alpha_{t,B})} }{ \alpha_{ min(\alpha_{s,A})}
            \alpha_{ min(\alpha_{t,B})} } \ln \varepsilon }

    where :math:`d` is the cut-off distance at which shells do not interact with each other.
    :math:`A` and `B` are the atoms each contraction are respectively centered on.
    :math: `\alpha` is the gaussian exponent
    :math: `s` and `t` index primitive Gaussian shells centered on atom `A` and `B` respectively.

    Parameters
    ----------
    contractions_one : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) associated with the first index of
        the integral.
    contractions_two : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) associated with the second index of
        the integral.
    tol_screen : float, optional
        The tolerance used for screening two-index integrals. The `tol_screen` is combined with the
        minimum contraction exponents to compute a cutoff which is compared to the distance between
        the contraction centers to decide whether the integral should be set to zero.

    Returns
    -------
    value : `bool`
        If integral should be screened, return `True`
    """

    # calculate distance cutoff
    alpha_a = min(contractions_one.exps)
    alpha_b = min(contractions_two.exps)
    r_12 = contractions_two.coord - contractions_one.coord
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(tol_screen))
    # integrals are screened if centers are further apart than the cutoff
    return np.linalg.norm(r_12) > cutoff


def get_points_mask_for_contraction(contractions, points, deriv_order, tol_screen):
    r"""Return a boolean mask indicating which points should be screened for a contraction shell

    A point is considered screened if it lies farther from the contraction center than a cutoff
    radius computed from the contraction parameters and the screening tolerance
    (see :func:`compute_primitive_cutoff_radius`).

    Parameters
    ----------
    contractions : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) for which the mask is computed.
    points : np.ndarray of shape (N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the
        basis functions are evaluated. Rows correspond to points; columns
        correspond to the :math:`x`, :math:`y`, and :math:`z` components.
    deriv_order : int
        Total order of the Cartesian derivative to consider (0 for the function itself).
    tol_screen : float
        Screening tolerance for excluding points. This value, together with the
        minimum contraction parameters, determines a cutoff distance. Points
        farther than the cutoff are excluded from contraction evaluation.

    Returns
    -------
    np.ndarray of bool, shape (N,)
        Boolean mask where `False` marks points to be screened out.
    """
    angm = contractions.angmom
    # reshape exponents for broadcasting
    exps = contractions.exps[:, np.newaxis]  # shape (K, 1)
    # use absolute value (indicating magnitude) of primitive contributions
    coeffs = np.abs(contractions.coeffs)  # shape (K, M)

    # compute cutoff radius for all primitives in all contractions
    r_cuts = compute_primitive_cutoff_radius(coeffs, exps, angm, deriv_order, tol_screen)

    # pick the maximum radius over all primitives and contractions
    cutoff_radius = np.max(r_cuts)

    # screen out points beyond the cutoff radius
    points_r = np.linalg.norm(points - contractions.coord, axis=1)
    return points_r <= cutoff_radius


def compute_primitive_cutoff_radius(c, alpha, angm, deriv_order, tol_screen):
    r"""Compute the cutoff radius for a primitive Gaussian or its derivatives.

    The cutoff radius is the maximum distance from the center of the primitive Gaussian at which
    the radial bound of the function or its Cartesian derivative remains above a given tolerance
    :math:`\epsilon`. This radius is computed by solving the equation:

    .. math::

        r^{2(\ell + k)} \chi(r)^2 = \epsilon^2

    where :math:`\chi(r)` is the radial part of the primitive Gaussian, defined as:

    .. math::

        \chi(r) = c \, n \, e^{-\alpha r^2}

    Here, :math:`c` is the coefficient of the primitive Gaussian, :math:`\alpha` is its exponent,
    :math:`\ell` is the angular momentum quantum number, :math:`k` is the total order of the
    derivative (0 for the function itself), and :math:`n` is the 3D normalization factor given by:

    .. math::

        n = \left( \frac{2 \alpha}{\pi} \right)^{\frac{3}{4}} (4 \alpha)^{\frac{\ell}{2}}

    The radial bound accounts for the polynomial factors arising from derivatives:

    .. math::

        |\partial_x^p \partial_y^q \partial_z^r \chi(\mathbf{r})| \lesssim (2\alpha)^k r^{\ell+k} e^{-\alpha r^2},
        \quad k = p+q+r

    Parameters
    ----------
    c : float
        Coefficient of the primitive Gaussian.
    alpha : float
        Exponent :math:`\alpha` of the primitive Gaussian.
    angm : int
        Angular momentum quantum number :math:`\ell` of the primitive Gaussian.
    deriv_order : int
        Total order :math:`k` of the Cartesian derivative to consider (0 for the function itself).
    tol_screen : float
        Radial bound tolerance :math:`\epsilon` for computing the cutoff radius.

    Returns
    -------
    float
        The cutoff radius at which the radial bound of the Gaussian (or its derivative) drops below
        the specified tolerance.
    """
    c = np.abs(c)
    # Compute 3D normalization factor n for the primitive Gaussian
    n = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (angm / 2)

    # effective angular momentum including derivative order
    eff_angm = angm + deriv_order
    # Worst-case k-th derivative scales as (2\alpha r)^k, so tighten tolerance by (2\alpha)^k
    eff_tol_screen = tol_screen / (2 * alpha) ** deriv_order

    # special case for effective angular momentum 0. Solution found using logarithm
    if eff_angm == 0:
        val_ratio = eff_tol_screen / (c * n)
        if np.all(val_ratio >= 1.0):
            return 0.0 if np.isscalar(val_ratio) else np.zeros_like(val_ratio)
        return np.sqrt(-np.log(val_ratio) / alpha)
    # general case for effective angular momentum > 0. Solution found in terms of the Lambert W
    # function W_{-1} branch corresponds to the outermost solution
    lambert_input_value = -2 * alpha * (eff_tol_screen / (c * n)) ** (2 / eff_angm) / eff_angm
    return np.sqrt(-(eff_angm / (2 * alpha)) * lambertw(lambert_input_value, k=-1).real)


# Maximum dimensionless product C(ax, kx) * C(ay, ky) * C(az, kz)
# over ax + ay + az = angm and kx + ky + kz = deriv_order,
# where C(a, k) = max_u | 2^a / sqrt((2a-1)!!) * d^k/du^k [u^a exp(-u^2)] |.
_DIMENSIONLESS_MAX_C = {
    (0, 0): 1.000000000000,
    (0, 1): 0.857763884961,
    (0, 2): 2.000000000000,
    (0, 3): 3.903566132919,
    (0, 4): 12.000000000000,
    (1, 0): 0.857763884961,
    (1, 1): 2.000000000000,
    (1, 2): 3.903566132919,
    (1, 3): 12.000000000000,
    (1, 4): 32.713910049089,
    (2, 0): 1.471517764686,
    (2, 1): 3.279330231535,
    (2, 2): 8.661458127143,
    (2, 3): 25.957555738331,
    (2, 4): 86.032054139669,
    (3, 0): 3.279330231535,
    (3, 1): 8.661458127143,
    (3, 2): 25.957555738331,
    (3, 3): 86.032054139669,
    (3, 4): 310.039792967788,
    (4, 0): 8.661458127143,
    (4, 1): 25.957555738331,
    (4, 2): 86.032054139669,
    (4, 3): 310.039792967788,
    (4, 4): 1200.333710212083,
    (5, 0): 25.957555738331,
    (5, 1): 86.032054139669,
    (5, 2): 310.039792967788,
    (5, 3): 1200.333710212083,
    (5, 4): 4947.674318615531,
    (6, 0): 86.032054139669,
    (6, 1): 310.039792967788,
    (6, 2): 1200.333710212083,
    (6, 3): 4947.674318615531,
    (6, 4): 21561.430397073495,
}


def _get_dimensionless_max_c(angm, deriv_order):
    """Return the maximum dimensionless envelope factor for (angm, deriv_order)."""
    if (angm, deriv_order) in _DIMENSIONLESS_MAX_C:
        return _DIMENSIONLESS_MAX_C[(angm, deriv_order)] * (1.0 + 1e-12)
    if deriv_order == 0:
        return 1.0 * (1.0 + 1e-12)
    # Safe upper bound for higher angular momentum or derivative orders:
    factor = 1.0
    for j in range(1, deriv_order + 1):
        factor *= 2.0 * (angm + j)
    return factor * (1.0 + 1e-12)


def compute_contraction_upper_bound(contractions, deriv_order):
    r"""Compute an upper bound for a contraction or its derivatives for any point.

    The upper bound for a contraction or its Cartesian derivative at any point is given by the
    sum of upper bounds of its constituent primitive Gaussians or their derivatives (see
    :func:`compute_primitive_upper_bound`).

    Parameters
    ----------
    contractions : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) for which the upper bound is computed.
    deriv_order : int
        Total order of the Cartesian derivative to consider (0 for the function itself).

    Returns
    -------
    float
        An upper bound for the absolute value of the contraction or its derivative at any point.
    """
    angm = contractions.angmom
    # reshape exponents for broadcasting
    exps = contractions.exps[:, np.newaxis]  # shape (K, 1)
    # use absolute value (indicating magnitude) of primitive contributions
    coeffs = np.abs(contractions.coeffs)  # shape (K, M)

    # compute upper bounds for all primitives in all contractions
    upper_bounds = compute_primitive_upper_bound(coeffs, exps, angm, deriv_order)

    # Sum over primitives (axis 0), and take the maximum over segmented contractions (axis 1)
    return float(np.max(np.sum(upper_bounds, axis=0)))


def compute_contraction_upper_bond(contractions, deriv_order):
    r"""Deprecated alias for :func:`compute_contraction_upper_bound`."""
    return compute_contraction_upper_bound(contractions, deriv_order)


def compute_primitive_upper_bound(c, alpha, angm, deriv_order):
    r"""Compute an upper bound for a primitive Gaussian or its derivatives at any point.

    The upper bound for a 3D Cartesian primitive Gaussian or its Cartesian derivative of total
    order :math:`k` at any point in space is bounded by:

    .. math::

        \max_{\mathbf{r}} |\partial^k g(\mathbf{r})| \le |c| \left(\frac{2\alpha}{\pi}\right)^{3/4}
        \alpha^{k/2} C(\ell, k)

    where :math:`c` is the contraction coefficient, :math:`\alpha` is the Gaussian exponent,
    :math:`\ell` is the angular momentum, and :math:`C(\ell, k)` is the maximum dimensionless
    envelope factor across all Cartesian angular momentum components and derivative partitions.

    Parameters
    ----------
    c : float or np.ndarray
        Coefficient of the primitive Gaussian.
    alpha : float or np.ndarray
        Exponent :math:`\alpha` of the primitive Gaussian.
    angm : int
        Angular momentum quantum number :math:`\ell` of the primitive Gaussian.
    deriv_order : int
        Total order :math:`k` of the Cartesian derivative to consider (0 for the function itself).

    Returns
    -------
    float or np.ndarray
        An upper bound for the absolute value of the primitive Gaussian or its derivative at any
        point.
    """
    prefactor = _get_dimensionless_max_c(angm, deriv_order)
    n_3d = (2.0 * alpha / np.pi) ** 0.75
    alpha_term = alpha ** (deriv_order / 2.0) if deriv_order > 0 else 1.0
    return np.abs(c) * n_3d * alpha_term * prefactor
