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


def get_points_mask_for_contraction(contractions, points, tol_screen):
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
    r_cuts = compute_primitive_cutoff_radius(coeffs, exps, angm, tol_screen)

    # pick the maximum radius over all primitives and contractions
    cutoff_radius = np.max(r_cuts)

    # screen out points beyond the cutoff radius
    points_r = np.linalg.norm(points - contractions.coord, axis=1)
    return points_r <= cutoff_radius


def compute_primitive_cutoff_radius(c, alpha, angm, tol_screen):
    r"""Compute the cutoff radius for a primitive Gaussian.

    The cutoff radius is the maximum distance from the center of the primitive Gaussian at which the
    radial density remains above a given tolerance :math:`\epsilon`. This radius is computed by
    solving the equation:

    .. math::

        r^{2\ell} \chi(r)^2 = \epsilon^2

    where :math:`\chi(r)` is the radial part of the primitive Gaussian, defined as:

    .. math::

        \chi(r) = c \, n \, e^{-\alpha r^2}

    Here, :math:`c` is the coefficient of the primitive Gaussian, :math:`\alpha` is its exponent,
    :math:`\ell` is the angular momentum quantum number, and :math:`n` is the normalization factor
    given by:

    .. math::

        n = \left( \frac{2 \alpha}{\pi} \right)^{\frac{1}{4}}
            \frac{(4 \alpha)^{\frac{\ell}{2}}}{\sqrt{(2\ell + 1)!!}}

    Parameters
    ----------
    c : float
        Coefficient of the primitive Gaussian.
    alpha : float
        Exponent :math:`\alpha` of the primitive Gaussian.
    angm : int
        Angular momentum quantum number :math:`\ell` of the primitive Gaussian.
    dens_tol : float
        Radial density tolerance :math:`\epsilon` for computing the cutoff radius.

    Returns
    -------
    float
        The cutoff radius for the primitive Gaussian.
    """
    # Compute normalization factor n for the primitive Gaussian
    n = (2 * alpha / np.pi) ** 0.25 * (4 * alpha) ** (angm / 2) / np.sqrt(factorial2(2 * angm + 1))
    # special case for angular momentum 0. Solution found using logarithm
    if angm == 0:
        return np.sqrt(-np.log(tol_screen / (c * n)) / alpha)
    # general case for angular momentum > 0. Solution found in terms of the Lambert W function
    # W_{-1} branch corresponds to the outermost solution
    lambert_input_value = -2 * alpha * (tol_screen / (c * n)) ** (2 / angm) / angm
    return np.sqrt(-(angm / (2 * alpha)) * lambertw(lambert_input_value, k=-1).real)
