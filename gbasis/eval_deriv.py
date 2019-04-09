"""Functions for evaluating Gaussian primitives."""
from gbasis.base_one import BaseOneIndex
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.deriv import _eval_deriv_contractions
import numpy as np


class EvalDeriv(BaseOneIndex):
    """Class for evaluating the Gaussian contractions and their linear combinations.

    The first dimension (axis 0) of the returned array is associated with a contracted Gaussian (or
    a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Contractions that are associated with each index of the array.
        Each tuple of ContractedCartesianGaussians corresponds to an index of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
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
        Return the evaluations of the Cartesian contractions of the instance at the given
        coordinates.
        `K_cart` is the total number of Cartesian contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_spherical(self, coords, orders) : np.ndarray(K_sph, N)
        Return the evaluations of the spherical contractions of the instance at the given
        coordinates.
        `K_sph` is the total number of spherical contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_spherical_lincomb(self, transform, coords, orders) : np.ndarray(K_orbs, N)
        Return the evaluations of the linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.
        `N` is the number of coordinates at which the contractions are evaluated.

    """

    @staticmethod
    def construct_array_contraction(contractions, coords, orders):
        """Return the array associated with a set of contracted Cartesian Gaussians.

        Parameters
        ----------
        contractions : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) that will be used to construct an
            array.
        coords : np.ndarray(N, 3)
            Points in space where the contractions are evaluated.
        orders : np.ndarray(3,)
            Orders of the derivative.

        Returns
        -------
        array_contraction : np.ndarray(M, L_cart, N)
            Array associated with the given instance(s) of ContractedCartesianGaussians.
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
            If contractions is not a ContractedCartesianGaussians instance.
            If coords is not a two-dimensional numpy array with 3 columns.
            If orders is not a one-dimensional numpy array with 3 elements.
        ValueError
            If orders has any negative numbers.
            If orders does not have dtype int.

        Note
        ----
        Since all of the keyword arguments of `construct_array_cartesian`,
        `construct_array_spherical`, and `construct_array_spherical_lincomb` are ultimately passed
        down to this method, all of the mentioned methods must be called with the keyword arguments
        `coords` and `orders`.

        """
        if not isinstance(contractions, ContractedCartesianGaussians):
            raise TypeError("`contractions` must be a ContractedCartesianGaussians instance.")
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
        angmom_comps = contractions.angmom_components
        center = contractions.coord
        norm = contractions.norm
        output = _eval_deriv_contractions(
            coords, orders, center, angmom_comps, alphas, prim_coeffs, norm
        )
        return output
