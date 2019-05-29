"""Functions for evaluating Gaussian primitives."""
from gbasis._deriv import _eval_deriv_contractions
from gbasis.base_one import BaseOneIndex
from gbasis.contractions import GeneralizedContractionShell
import numpy as np


class EvalDeriv(BaseOneIndex):
    """Class for evaluating the Gaussian contractions and their linear combinations.

    The first dimension (axis 0) of the returned array is associated with a contracted Gaussian (or
    a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Contractions that are associated with each index of the array.
        Each tuple of GeneralizedContractionShell corresponds to an index of the array.

    Properties
    ----------
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first index of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contraction, coords, orders) : np.ndarray(M, L_cart, N)
        Return the evaluations of the given Cartesian contractions at the given coordinates.
        `M` is the number of segmented contractions with the same exponents (and angular momentum).
        `L_cart` is the number of Cartesian contractions for the given angular momentum.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_cartesian(self, coords, orders) : np.ndarray(K_cart, N)
        Return the evaluations of the derivatives of the Cartesian contractions of the instance at
        the given coordinates.
        `K_cart` is the total number of Cartesian contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_spherical(self, coords, orders) : np.ndarray(K_sph, N)
        Return the evaluations of the derivatives of the spherical contractions of the instance at
        the given coordinates.
        `K_sph` is the total number of spherical contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_mix(self, coord_types, coords, orders) : np.ndarray(K_cont, N)
        Return the array associated with all of the contraction in the given coordinate system.
        `K_cont` is the total number of contractions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_lincomb(self, transform, coord_type, coords, orders) : np.ndarray(K_orbs, N)
        Return the evaluatiosn of derivatives of the  linear combinations of contractions in the
        given coordinate system.
        `K_orbs` is the number of basis functions produced after the linear combinations.
        `N` is the number of coordinates at which the contractions are evaluated.

    """

    @staticmethod
    def construct_array_contraction(contractions, coords, orders):
        """Return the array associated with a set of contracted Cartesian Gaussians.

        Parameters
        ----------
        contractions : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) that will be used to construct an
            array.
        coords : np.ndarray(N, 3)
            Points in space where the contractions are evaluated.
        orders : np.ndarray(3,)
            Orders of the derivative.

        Returns
        -------
        array_contraction : np.ndarray(M, L_cart, N)
            Array associated with the given instance(s) of GeneralizedContractionShell.
            First index corresponds to segmented contractions within the given generalized
            contraction (same exponents and angular momentum, but different coefficients). `M` is
            the number of segmented contractions with the same exponents (and angular momentum).
            Second index corresponds to angular momentum vector. `L_cart` is the number of Cartesian
            contractions for the given angular momentum.
            Third index corresponds to coordinates at which the contractions are evaluated. `N` is
            the number of coordinates at which the contractions are evaluated.

        Raises
        ------
        TypeError
            If contractions is not a GeneralizedContractionShell instance.
            If coords is not a two-dimensional numpy array with 3 columns.
            If orders is not a one-dimensional numpy array with 3 elements.
        ValueError
            If orders has any negative numbers.
            If orders does not have dtype int.

        Note
        ----
        Since all of the keyword arguments of `construct_array_cartesian`,
        `construct_array_spherical`, and `construct_array_lincomb` are ultimately passed
        down to this method, all of the mentioned methods must be called with the keyword arguments
        `coords` and `orders`.

        """
        if not isinstance(contractions, GeneralizedContractionShell):
            raise TypeError("`contractions` must be a GeneralizedContractionShell instance.")
        if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
            raise TypeError(
                "`coords` must be given as a two-dimensional numpy array with 3 columnms."
            )
        if not (isinstance(orders, np.ndarray) and orders.shape == (3,)):
            raise TypeError(
                "Orders of the derivatives must be a one-dimensional numpy array with 3 elements."
            )
        if np.any(orders < 0):
            raise ValueError("Negative order of derivative is not supported.")
        if orders.dtype != int:
            raise ValueError("Orders of the derivatives must be given as integers.")

        alphas = contractions.exps
        prim_coeffs = contractions.coeffs
        angmom_comps = contractions.angmom_components_cart
        center = contractions.coord
        norm_prim = contractions.norm_prim
        output = _eval_deriv_contractions(
            coords, orders, center, angmom_comps, alphas, prim_coeffs, norm_prim
        )
        return output


def evaluate_deriv_basis_cartesian(basis, coords, orders):
    """Evaluate the derivatives of a basis set in the Cartesian form at the given coordinates.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.

    Returns
    -------
    eval_array : np.ndarray(K_cart, N)
        Evaluations of the derivatives of the Cartesian contractions of the instance at the given
        coordinates.
        `K_cart` is the total number of Cartesian contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    return EvalDeriv(basis).construct_array_cartesian(coords=coords, orders=orders)


def evaluate_deriv_basis_spherical(basis, coords, orders):
    """Evaluate the derivatives of a basis set in the spherical form at the given coordinates.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.

    Returns
    -------
    eval_array : np.ndarray(K_sph, N)
        Evaluations of the derivatives of the spherical contractions of the instance at the given
        coordinates.
        `K_sph` is the total number of spherical contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    return EvalDeriv(basis).construct_array_spherical(coords=coords, orders=orders)


def evaluate_deriv_basis_mix(basis, coords, orders, coord_types):
    """Evaluate derivatives of basis set in the given coordinate system at the given coordinates.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
    coord_types : list/tuple of str
        Types of the coordinate system for each GeneralizedContractionShell.
        Each entry must be one of "cartesian" or "spherical".

    Returns
    -------
    eval_array : np.ndarray(K_cont, N)
        Evaluations of the derivatives of the contractions of the given coordinate types at the
        given coordinates.
        `K_cont` is the total number of contractions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    return EvalDeriv(basis).construct_array_mix(coord_types, coords=coords, orders=orders)


def evaluate_deriv_basis_lincomb(basis, coords, orders, transform, coord_type="spherical"):
    """Evaluate a basis set in the spherical form at the given coordinates.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from contractions in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left.
        Rows correspond to the linear combinationes (i.e. MO) and the columns correspond to the
        contractions (i.e. AO).
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    eval_array : np.ndarray(K_orbs, N)
        Evaluations of derivatives of the linear combinations of contractions.
        `K_orbs` is the number of basis functions produced after the linear combinations.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    return EvalDeriv(basis).construct_array_lincomb(
        transform, coord_type, coords=coords, orders=orders
    )
