"""1 and 2-index screening functions"""

import numpy as np


def two_index_screening(contractions_one, contractions_two, tol_screen):
    r"""Return True or False in response to whether integral is required.

    .. math::
           d_{A_s;B_t} > \sqrt{-\frac{\alpha_{ min(\alpha_{s,A})} +
           \alpha_{ min(\alpha_{t,B})} }{ \alpha_{ min(\alpha_{s,A})}
            \alpha_{ min(\alpha_{t,B})} } \ln \varepsilon }

    where :math:`d` is the cut-off distance at which shells do not interact with each other.
    :math:`A` and `B` are the atoms each contraction are respectively centered on.
    :math: `\alpha` is the gaussian exponent
    :math: `s` and `t` index the primitive Gaussian shells centered on atom `A` and `B` respectively.

    Parameters
    ----------
    contractions_one : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) associated with the first index of
        the integral.
    contractions_two : GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) associated with the second index of
        the integral.
    tol_screen : bool or float, optional
        The tolerance used for screening two-index integrals. The `tol_screen` is combined with the
        minimum contraction exponents to compute a cutoff which is compared to the distance between
        the contraction centers to decide whether the integral should be set to zero.
        If `None`, no screening is performed.

    Returns
    -------
    value : `bool`
        If integral is required, return `True`
    """

    # check screening_tol
    if isinstance(tol_screen, bool):
        raise TypeError(f"Argument tol_screen must be a float or None. Got {type(tol_screen)}.")

    # test if screening is required
    if tol_screen is None:
        return False

    # calculate distance cutoff
    alpha_a = min(contractions_one.exps)
    alpha_b = min(contractions_two.exps)
    r_12 = contractions_two.coord - contractions_one.coord
    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * np.log(tol_screen))
    return np.linalg.norm(r_12) > cutoff
