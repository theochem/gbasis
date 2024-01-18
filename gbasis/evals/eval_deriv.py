"""Functions for evaluating Gaussian primitives."""
from gbasis.base_one import BaseOneIndex
from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import _eval_deriv_contractions
from gbasis.evals._deriv import _eval_first_second_order_deriv_contractions
import numpy as np


class EvalDeriv(BaseOneIndex):
    """Class for evaluating Gaussian contractions and their linear combinations.

    Dimension 0 of the returned array is associated with a contracted Gaussian (or
    a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Contractions that are associated with each index of the array.
        Each tuple of `GeneralizedContractionShell` corresponds to an index of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first index of the array.
        Property of `EvalDeriv`.


    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contraction, points, orders) : np.ndarray(M, L_cart, N)
        Return the evaluations of the given Cartesian contractions at the given coordinates.
        `M` is the number of segmented contractions with the same exponents (and angular
        momentum).
        `L_cart` is the number of Cartesian contractions for the given angular momentum.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_cartesian(self, points, orders) : np.ndarray(K_cart, N)
        Return the evaluations of the derivatives of the Cartesian contractions of the instance at
        the given coordinates.
        `K_cart` is the total number of Cartesian contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_spherical(self, points, orders) : np.ndarray(K_sph, N)
        Return the evaluations of the derivatives of the spherical contractions of the instance at
        the given coordinates.
        `K_sph` is the total number of spherical contractions within the instance.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_mix(self, coord_types, points, orders) : np.ndarray(K_cont, N)
        Return the array associated with all of the contraction in the given coordinate system.
        `K_cont` is the total number of contractions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.
    construct_array_lincomb(self, transform, coord_type, points, orders) : np.ndarray(K_orbs, N)
        Return the evaluation of derivatives of the  linear combinations of contractions in the
        given coordinate system.
        `K_orbs` is the number of basis functions produced after the linear combinations.
        `N` is the number of coordinates at which the contractions are evaluated.

    """

    @staticmethod
    def construct_array_contraction(contractions, points, orders, deriv_type="general"):
        r"""Return the array associated with a set of contracted Cartesian Gaussians.

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
        orders : np.ndarray(3,)
            Orders of the derivative.
        deriv_type : "general" or "direct"
            Specification of derivative of contraction function in _deriv.py. "general"
            makes reference to general implementation of any order derivative
            function (_eval_deriv_contractions()) and "direct" makes reference to specific
            implementation of first and second order derivatives for generalized
            contraction (_eval_first_second_order_deriv_contractions()).

        Returns
        -------
        array_contraction : np.ndarray(M, L_cart, N)
            Array associated with the given instance(s) of GeneralizedContractionShell.
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
            If orders is not a one-dimensional `numpy` array with 3 elements.
        ValueError
            If orders has any negative numbers.
            If orders does not have `dtype` int.

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
        if not (isinstance(orders, np.ndarray) and orders.shape == (3,)):
            raise TypeError(
                "Orders of the derivatives must be a one-dimensional `numpy` array with 3 elements."
            )
        if np.any(orders < 0):
            raise ValueError("Negative order of derivative is not supported.")
        if orders.dtype != int:
            raise ValueError("Orders of the derivatives must be given as integers.")

        alphas = contractions.exps
        prim_coeffs = contractions.coeffs
        angmom_comps = contractions.angmom_components_cart
        center = contractions.coord
        norm_prim_cart = contractions.norm_prim_cart
        if deriv_type == "general":
            output = _eval_deriv_contractions(
                points, orders, center, angmom_comps, alphas, prim_coeffs, norm_prim_cart
            )
        elif deriv_type == "direct":
            output = _eval_first_second_order_deriv_contractions(
                points, orders, center, angmom_comps, alphas, prim_coeffs, norm_prim_cart
            )
        return output


def evaluate_deriv_basis(
    basis,
    points,
    orders,
    transform=None,
    deriv_type="general",
):
    r"""Evaluate the derivative of the basis set in the given coordinate system at the given points.

    The derivative (to arbitrary orders) of a basis function is given by:

    .. math::
    
        \frac{\partial^{m_x + m_y + m_z}}{\partial x^{m_x} \partial y^{m_y} \partial z^{m_z}}
        \phi (\mathbf{r})
    
    where :math:`m_x, m_y, m_z` are the orders of the derivative with respect to x, y, and z, 
    :math:`\phi` is the basis function (a generalized contraction shell), and :math:`\mathbf{r}_n` are
    the coordinate of the points in space where the basis function is evaluated.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    orders : np.ndarray(3,)
        Orders of the derivative.
        First element corresponds to the order of the derivative with respect to x.
        Second element corresponds to the order of the derivative with respect to y.
        Thirds element corresponds to the order of the derivative with respect to z.
    transform : np.ndarray(K, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).

    Returns
    -------
    eval_array : np.ndarray(K, N)
        Evaluations of the derivative of the basis functions at the given points.
        If keyword argument `transform` is provided, then the transformed basis functions will be
        evaluated at the given points.
        `K` is the total number of basis functions within the given basis set.
        `N` is the number of coordinates at which the contractions are evaluated.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if transform is not None:
        return EvalDeriv(basis).construct_array_lincomb(
            transform, coord_type, points=points, orders=orders, deriv_type=deriv_type
        )
    if all(ct == "cartesian" for ct in coord_type) or coord_type == "cartesian":
        return EvalDeriv(basis).construct_array_cartesian(
            points=points, orders=orders, deriv_type=deriv_type
        )
    if all(ct == "spherical" for ct in coord_type) or coord_type == "spherical":
        return EvalDeriv(basis).construct_array_spherical(
            points=points, orders=orders, deriv_type=deriv_type
        )
    return EvalDeriv(basis).construct_array_mix(
        coord_type, points=points, orders=orders, deriv_type=deriv_type
    )
