"""1 and 2-index screening functions"""

import numpy as np
from scipy.special import factorial2, lambertw


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


def evaluate_basis_mask(basis, points, tol_screen):
    """
    Compute a masks indicating which grid points are within the
    effective cutoff radius of each contracted Gaussian basis function.
    This function calculates a distance cutoff for each contracted basis
    function in `basis` such that contributions below `tol_screen`
    are neglected. For each contraction, the most diffuse primitive
    is identified, and the cutoff is determined analytically.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    tol_screen : float, optional
        The tolerance used for screening one-index evaluations. `tol_screen` is combined with the
        most diffuse primitive parameters to compute a cutoff, which is compared to the distance
        between the contraction center to determine whether the evaluation should be set to zero.

    Returns
    -------
    mask : list of ndarray of shape (N,)
        A list of boolean arrays, one for each contraction in `basis`.
        Each array marks with `True` the points within the cutoff radius
        for that contraction and `False` otherwise.
    """
    mask = []
    for contraction in basis:
        ## most diffuse primitive
        index = np.argmin(contraction.exps)
        cmin = np.abs(contraction.coeffs[index])
        amin = contraction.exps[index]
        angm = contraction.angmom
        nmin = (
            (2 * amin / np.pi) ** (3 / 4)
            * ((4 * amin) ** (angm / 2))
            / np.sqrt(factorial2(2 * angm + 1))
        )
        ## log formula for l=0
        if angm == 0:
            cutoff = np.sqrt(-np.log(tol_screen / (cmin * nmin)) / amin)
        ## lambert formula otherwise
        else:
            w = lambertw(
                (-(2.0 * amin / angm) * ((tol_screen / (cmin * nmin)) ** (2.0 / angm))), k=-1
            ).real
            cutoff = np.sqrt(-(angm / (2.0 * amin)) * w)
        distance = np.linalg.norm(points - contraction.coord, axis=1)
        mask.append(distance <= cutoff)
    return mask
