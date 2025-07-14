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
    tol_screen : float
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


def evaluate_contraction_mask(contraction, points, tol_screen):
    r"""Return a boolean mask of points that should be screened.

    .. math::

        d =
            \begin{cases}
                \sqrt{ -\dfrac{\ln \left( \dfrac{\epsilon}{c_{\min} \times \alpha_{\min}} \right)}
		{\alpha_{\min}} }, & \text{if } \ell = 0 \\
                \sqrt{ -\dfrac{W_{-1}
		\left( \dfrac{\epsilon}{c_{\min} \times \alpha_{\min}} \right)}
		{\alpha_{\min}} }, & \text{otherwise}
            \end{cases}

    where :math:`d` is the cutoff distance beyond which the
    contraction does not interact with a grid point.

    :math:`\alpha_{\min}` is the Gaussian exponent
    :math:`c_{\min}` is the Gaussian coefficient, and :math:`n_{\min}` is given by:

    .. math::
        n_{\min} =
            \left( \dfrac{2 \alpha_{\min}}{\pi} \right)^{3/4}
            \cdot \dfrac{(4 \alpha_{\min})^{\ell / 2}}{\sqrt{(2\ell + 1)!!}}

    Parameters
    ----------
    contraction : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) associated with the first index of
        the integral.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis
        functions are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    tol_screen : float
        The tolerance used for screening a contraction at grid points. `tol_screen` is combined
        with the minimum contraction parameters to compute a cutoff distance. This cutoff is
        compared against all grid points, point farther than the cutoff will be excluded
        from evaluation of the contraction.

    Returns
    -------
    array : `bool` (N, 3)
        For each grid point, if evaluation should be screened, return `False`
    """

    # minimum contraction parameters
    cmin = np.abs(contraction.coeffs.min())
    amin = contraction.exps.min()
    angm = contraction.angmom
    nmin = (
        (2 * amin / np.pi) ** (3 / 4)
        * ((4 * amin) ** (angm / 2))
        / np.sqrt(factorial2(2 * angm + 1))
    )

    # log formula for l = 0
    if angm == 0:
        dist2 = -np.log(tol_screen / (cmin * nmin)) / amin
    # lambert formula otheriwse
    else:
        w = lambertw((tol_screen / (cmin * nmin)), k=-1)
        x = -w.real / amin
        dist2 = x

    return np.sum((points - contraction.coord) ** 2, axis=1) <= dist2
