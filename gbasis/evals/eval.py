"""Functions for evaluating Gaussian contractions."""

from gbasis.base_one import BaseOneIndex
from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.screening import get_points_mask_for_contraction
import numpy as np


class Eval(BaseOneIndex):
    """Class for evaluating Gaussian contractions and their linear combinations.

    Dimension 0 of the returned array is associated with a contracted Gaussian (or
    a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Contractions that are associated with each index of the array.
        Each tuple of GeneralizedContractionShell corresponds to an index of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first index of the array.
        Property of `Eval`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contraction, points) : np.ndarray(M, L_cart, N)
        Return the evaluations of the given Cartesian contractions at the given coordinates.
        `M` is the number of segmented contractions with the same exponents (and angular
        momentum).
        `L_cart` is the number of Cartesian contractions for the given angular momentum.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_cartesian(self, points) : np.ndarray(K_cart, N)
        Return the evaluations of the Cartesian contractions of the instance at the given
        coordinates.
        `K_cart` is the total number of Cartesian contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_spherical(self, points) : np.ndarray(K_sph, N)
        Return the evaluations of the spherical contractions of the instance at the given
        coordinates.
        `K_sph` is the total number of spherical contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_mix(self, coord_types, points) : np.ndarray(K_cont, N)
        Return the evaluatations of the contraction in the given coordinate system.
        `K_cont` is the total number of contractions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_lincomb(self, transform, coord_type, points) : np.ndarray(K_orbs, N)
        Return the evaluations of the linear combinations of contractions in the given coordinate
        system.
        `K_orbs` is the number of basis functions produced after the linear combinations.
        `N` is the number of coordinates at which the contractions are evaluated.

    """

    @staticmethod
    def construct_array_contraction(contractions, points, screen_basis=False, tol_screen=1e-8):
        r"""Return the evaluations of the given contractions at the given coordinates.

        Parameters
        ----------
        contractions : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) that will be used to construct an
            array.
        points : np.ndarray(N, 3)
            Cartesian coordinates of the points in space (in atomic units) where the basis
            functions are evaluated.
            Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
            components.
        screen_basis : bool, optional
            Whether to screen out points with negligible contributions. Default value is False.
        tol_screen : float
            Screening tolerance for excluding evaluations. Points with values below this tolerance
            will not be evaluated (they will be set to zero). Internal computed quantities that
            affect the results below this tolerance will also be ignored to speed up the
            evaluation. Default value is 1e-8.

        Returns
        -------
        array_contraction : np.ndarray(M, L_cart, N)
            Evaluations of the given Cartesian contractions at the given points.
            Dimension 0 corresponds to segmented contractions within the given generalized
            contraction (same exponents and angular momentum, but different coefficients). `M` is
            the number of segmented contractions with the same exponents (and angular momentum).
            Dimension 1 corresponds to angular momentum vector. `L_cart` is the number of Cartesian
            contractions for the given angular momentum.
            Dimension 2 corresponds to coordinates at which the contractions are evaluated. `N` is
            the number of coordinates at which the contractions are evaluated.

        Raises
        ------
        TypeError
            If contractions is not a `GeneralizedContractionShell` instance.
            If points is not a two-dimensional `numpy` array with 3 columns.

        Note
        ----
        Since all of the keyword arguments of `construct_array_cartesian`,
        `construct_array_spherical`, and `construct_array_lincomb` are ultimately passed
        down to this method, all of the mentioned methods must be called with the keyword arguments
        `points` and `orders`.

        """
        if not isinstance(contractions, GeneralizedContractionShell):
            raise TypeError("`contractions` must be a `GeneralizedContractionShell` instance.")
        if not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3):
            raise TypeError(
                "`points` must be given as a two-dimensional `numpy` array with 3 columns."
            )

        alphas = contractions.exps
        prim_coeffs = contractions.coeffs
        angmom_comps = contractions.angmom_components_cart
        center = contractions.coord
        norm_prim_cart = contractions.norm_prim_cart

        # if screening is not requested, evaluate all points
        if not screen_basis:
            return _eval_deriv_contractions(
                points, np.zeros(3), center, angmom_comps, alphas, prim_coeffs, norm_prim_cart
            )

        # default case, screen points that are too far from the contraction center
        pts_mask = get_points_mask_for_contraction(
            contractions, points, deriv_order=0, tol_screen=tol_screen
        )
        # reconstruct the array with correct shape
        L = angmom_comps.shape[0]
        M = prim_coeffs.shape[1]
        N = points.shape[0]
        output = np.zeros((M, L, N), dtype=np.float64)

        # fill non-screened points in the output array
        output[:, :, pts_mask] = _eval_deriv_contractions(
            points[pts_mask],
            np.zeros(3),
            center,
            angmom_comps,
            alphas,
            prim_coeffs,
            norm_prim_cart,
        )
        return output


def evaluate_basis(basis, points, transform=None, screen_basis=False, tol_screen=1e-8):
    r"""Evaluate the basis set in the given coordinate system at the given points.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis
        functions are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    eval_array : np.ndarray(K, N)
        Evaluations of the basis functions at the given points.
        If keyword argument `transform` is provided, then the transformed basis functions will be
        evaluated at the given points.
        `K` is the total number of basis functions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]
    kwargs = {"tol_screen": tol_screen, "screen_basis": screen_basis}

    if transform is not None:
        return Eval(basis).construct_array_lincomb(transform, coord_type, points=points, **kwargs)
    if all(ct == "cartesian" for ct in coord_type):
        return Eval(basis).construct_array_cartesian(points=points, **kwargs)
    if all(ct == "spherical" for ct in coord_type):
        return Eval(basis).construct_array_spherical(points=points, **kwargs)
    return Eval(basis).construct_array_mix(coord_type, points=points, **kwargs)
