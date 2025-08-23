"""Functions for evaluating Gaussian primitives."""

import numpy as np

from gbasis.base_one import BaseOneIndex
from gbasis.contractions import GeneralizedContractionShell
from gbasis.evals._deriv import (
    _eval_deriv_contractions,
    _eval_first_second_order_deriv_contractions,
)
from gbasis.screening import evaluate_basis_mask
from gbasis.spherical import generate_transformation


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

    def construct_array_cartesian(self, points, orders, deriv_type, mask):
        """Return the array associated with the given set of contracted Cartesian Gaussians.

        Parameters
        ----------
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
        deriv_type : "general" or "direct"
            Specification of derivative of contraction function in _deriv.py. "general"
            makes reference to general implementation of any order derivative
            function (_eval_deriv_contractions()) and "direct" makes reference to specific
            implementation of first and second order derivatives for generalized
            contraction (_eval_first_second_order_deriv_contractions()).
        mask : list of ndarray of shape (N,)
            A list of boolean arrays, one for each contraction in `basis`.
            Each array marks with `True` the points within the cutoff radius
            for that contraction and `False` otherwise.

        Returns
        -------
        array : np.ndarray(K_cart, ...)
            Array associated with the given set of contracted Cartesian Gaussians.
            Dimension 0 is associated with the contracted Cartesian Gaussian. `K_cart` is the
            total number of Cartesian contractions within the instance.

        """
        matrices = []
        for i, contraction in enumerate(self.contractions):
            if mask is not None:
                points_subset = points[mask[i]]
                subset_array = self.construct_array_contraction(
                    contraction, points_subset, orders=orders, deriv_type=deriv_type
                )
                array = np.zeros((subset_array.shape[0], subset_array.shape[1], len(points)))
                array[:, :, mask[i]] = subset_array
            else:
                array = self.construct_array_contraction(
                    contraction, points, orders=orders, deriv_type=deriv_type
                )
            # array = self.construct_array_contraction(contraction, **kwargs)
            # normalize contractions
            array *= contraction.norm_cont.reshape(*array.shape[:2], *[1 for _ in array.shape[2:]])
            # ASSUME array always has shape (M, L, ...)
            matrices.append(np.concatenate(array, axis=0))
        return np.concatenate(matrices, axis=0)

    def construct_array_spherical(self, points, orders, deriv_type, mask):
        """Return the array associated with contracted spherical Gaussians (atomic orbitals).

        Parameters
        ----------
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
        deriv_type : "general" or "direct"
            Specification of derivative of contraction function in _deriv.py. "general"
            makes reference to general implementation of any order derivative
            function (_eval_deriv_contractions()) and "direct" makes reference to specific
            implementation of first and second order derivatives for generalized
            contraction (_eval_first_second_order_deriv_contractions()).
        mask : list of ndarray of shape (N,)
            A list of boolean arrays, one for each contraction in `basis`.
            Each array marks with `True` the points within the cutoff radius
            for that contraction and `False` otherwise.

        Returns
        -------
        array : np.ndarray(K_sph, ...)
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.
            Dimension 0 is associated with the contracted spherical Gaussian. `K_sph` is the
            total number of Cartesian contractions within the instance.

        """
        matrices_spherical = []
        for i, cont in enumerate(self.contractions):
            # get transformation from cartesian to spherical (applied to left)
            transform = generate_transformation(
                cont.angmom, cont.angmom_components_cart, cont.angmom_components_sph, "left"
            )
            if mask is not None:
                # evaluate the function at the given points
                points_subset = points[mask[i]]
                subset_matrix_contraction = self.construct_array_contraction(
                    cont, points_subset, orders=orders, deriv_type=deriv_type
                )
                matrix_contraction = np.zeros(
                    (
                        subset_matrix_contraction.shape[0],
                        subset_matrix_contraction.shape[1],
                        len(points),
                    )
                )
                matrix_contraction[:, :, mask[i]] = subset_matrix_contraction
            else:
                matrix_contraction = self.construct_array_contraction(
                    cont, points, orders=orders, deriv_type=deriv_type
                )
            # evaluate the function at the given points
            ##matrix_contraction = self.construct_array_contraction(cont, **kwargs)
            # normalize contractions
            matrix_contraction *= cont.norm_cont.reshape(
                *matrix_contraction.shape[:2], *[1 for _ in matrix_contraction.shape[2:]]
            )
            # transform
            # ASSUME array always has shape (M, L, ...)
            matrix_contraction = np.tensordot(transform, matrix_contraction, (1, 1))
            matrix_contraction = np.concatenate(np.swapaxes(matrix_contraction, 0, 1), axis=0)
            # store
            matrices_spherical.append(matrix_contraction)

        return np.concatenate(matrices_spherical, axis=0)

    def construct_array_mix(self, coord_types, points, orders, deriv_type, mask):
        """Return the array associated with all of the contractions in the given coordinate system.

        Parameters
        ----------
        coord_types : list/tuple of str
            Types of the coordinate system for each GeneralizedContractionShell.
            Each entry must be one of "cartesian" or "spherical".
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
        deriv_type : "general" or "direct"
            Specification of derivative of contraction function in _deriv.py. "general"
            makes reference to general implementation of any order derivative
            function (_eval_deriv_contractions()) and "direct" makes reference to specific
            implementation of first and second order derivatives for generalized
            contraction (_eval_first_second_order_deriv_contractions()).
        mask : list of ndarray of shape (N,)
            A list of boolean arrays, one for each contraction in `basis`.
            Each array marks with `True` the points within the cutoff radius
            for that contraction and `False` otherwise.

        Returns
        -------
        array : np.ndarray(K_cont, ...)
            Array associated with the spherical contrations of the basis set.
            Dimension 0 is associated with each spherical contraction in the basis set.
            `K_cont` is the total number of contractions within the given basis set.

        Raises
        ------
        TypeError
            If `coord_types` is not a list/tuple.
        ValueError
            If `coord_types` has an entry that is not "cartesian" or "spherical".
            If `coord_types` has different number of entries as the number of
            `GeneralizedContractionShell` (`contractions`) in instance.

        """
        if not isinstance(coord_types, (list, tuple)):
            raise TypeError("`coord_types` must be a list or a tuple.")
        if not all(i in ["cartesian", "spherical"] for i in coord_types):
            raise ValueError(
                "Each entry of `coord_types` must be one of 'cartesian' or 'spherical'."
            )
        if len(coord_types) != len(self.contractions):
            raise ValueError(
                "`coord_types` must have the same number of entries as the number of "
                "`GeneralizedContractionShell` in the instance."
            )

        matrices = []
        for i, (cont, coord_type) in enumerate(zip(self.contractions, coord_types)):
            if mask is not None:
                # evaluate the function at the given points
                points_subset = points[mask[i]]
                subset_matrix_contraction = self.construct_array_contraction(
                    cont, points_subset, orders=orders, deriv_type=deriv_type
                )
                matrix_contraction = np.zeros(
                    (
                        subset_matrix_contraction.shape[0],
                        subset_matrix_contraction.shape[1],
                        len(points),
                    )
                )
                matrix_contraction[:, :, mask[i]] = subset_matrix_contraction
            else:
                matrix_contraction = self.construct_array_contraction(
                    cont, points, orders=orders, deriv_type=deriv_type
                )
            # normalize contractions
            matrix_contraction *= cont.norm_cont.reshape(
                *matrix_contraction.shape[:2], *[1 for _ in matrix_contraction.shape[2:]]
            )
            if coord_type == "spherical":
                # get transformation from cartesian to spherical
                # (applied to left), only when it is needed.
                transform = generate_transformation(
                    cont.angmom, cont.angmom_components_cart, cont.angmom_components_sph, "left"
                )
                # Apply the transform.
                # ASSUME array always has shape (M, L, ...)
                matrix_contraction = np.tensordot(transform, matrix_contraction, (1, 1))
                matrix_contraction = np.swapaxes(matrix_contraction, 0, 1)
            matrix_contraction = np.concatenate(matrix_contraction, axis=0)
            # store
            matrices.append(matrix_contraction)

        return np.concatenate(matrices, axis=0)

    def construct_array_lincomb(self, transform, coord_type, points, orders, deriv_type, mask):
        r"""Return the array associated with linear combinations of contractions.

        .. math::

            \sum_{j} T_{i j} M_{jklm...} = M^{trans}_{iklm...}

        Parameters
        ----------
        transform : np.ndarray(K_orbs, K_cont)
            Transformation matrix from contractions in the given coordinate system (e.g. AO) to
            linear combinations of contractions (e.g. MO).
            Transformation is applied to the left.
            Rows correspond to the linear combinationes (i.e. MO) and the columns correspond to the
            contractions (i.e. AO).
        coord_type : list/tuple of str
            Types of the coordinate system for each GeneralizedContractionShell.
            Each entry must be one of "cartesian" or "spherical". If multiple
            instances of GeneralizedContractionShell are given but only one string
            ("cartesian" or "spherical") is provided in the list/tuple, all of the
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
        deriv_type : "general" or "direct"
            Specification of derivative of contraction function in _deriv.py. "general"
            makes reference to general implementation of any order derivative
            function (_eval_deriv_contractions()) and "direct" makes reference to specific
            implementation of first and second order derivatives for generalized
            contraction (_eval_first_second_order_deriv_contractions()).
        mask : list of ndarray of shape (N,)
            A list of boolean arrays, one for each contraction in `basis`.
            Each array marks with `True` the points within the cutoff radius
            for that contraction and `False` otherwise.

        Returns
        -------
        array : np.ndarray(K_orbs, ...)
            Array whose first index is associated with the linear combinations of the contractions.
            `K_orbs` is the number of basis functions produced after the linear combinations.

        Raises
        ------
        TypeError
            If `coord_type` is not a list/tuple of the strings 'cartesian' or 'spherical'.

        """
        if all(ct == "cartesian" for ct in coord_type):
            array = self.construct_array_cartesian(
                points=points, orders=orders, deriv_type=deriv_type, mask=mask
            )
        elif all(ct == "spherical" for ct in coord_type):
            array = self.construct_array_spherical(
                points=points, orders=orders, deriv_type=deriv_type, mask=mask
            )
        elif isinstance(coord_type, (list, tuple)):
            array = self.construct_array_mix(
                coord_type, points=points, orders=orders, deriv_type=deriv_type, mask=mask
            )
        else:
            raise TypeError(
                "`coord_type` must be a list/tuple of the strings 'cartesian' or 'spherical'"
            )
        return np.tensordot(transform, array, (1, 0))


def evaluate_deriv_basis(
    basis,
    points,
    orders,
    transform=None,
    deriv_type="general",
    screen_basis=True,
    tol_screen=1e-8,
):
    r"""Evaluate the derivative of the basis set in the given coordinate system at the given points.

    The derivative (to arbitrary orders) of a basis function is given by:

    .. math::

        \frac{\partial^{m_x + m_y + m_z}}{\partial x^{m_x} \partial y^{m_y} \partial z^{m_z}}
        \phi (\mathbf{r})

    where :math:`m_x, m_y, m_z` are the orders of the derivative with respect to x, y, and z,
    :math:`\phi` is the basis function (a generalized contraction shell), and :math:`\mathbf{r}_n`
    are the coordinate of the points in space where the basis function is evaluated.

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
        Specification of derivative of contraction function in _deriv.py. "general"
        makes reference to general implementation of any order derivative
        function (_eval_deriv_contractions()) and "direct" makes reference to specific
        implementation of first and second order derivatives for generalized
        contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        A toggle to enable or disable screening. Default value is `True` to enable screening.
    tol_screen : float, optional
        The tolerance used for screening one-index evaluations. `tol_screen` is combined with the
        most diffuse primitive parameters to compute a cutoff, which is compared to the distance
        between the contraction center to determine whether the evaluation should be set to zero.
        The default value for `tol_screen` is 1e-8.

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
    if screen_basis:
        mask = evaluate_basis_mask(basis, points, tol_screen)
    else:
        mask = None

    if transform is not None:
        return EvalDeriv(basis).construct_array_lincomb(
            transform, coord_type, points=points, orders=orders, deriv_type=deriv_type, mask=mask
        )
    if all(ct == "cartesian" for ct in coord_type) or coord_type == "cartesian":
        return EvalDeriv(basis).construct_array_cartesian(
            points=points, orders=orders, deriv_type=deriv_type, mask=mask
        )
    if all(ct == "spherical" for ct in coord_type) or coord_type == "spherical":
        return EvalDeriv(basis).construct_array_spherical(
            points=points, orders=orders, deriv_type=deriv_type, mask=mask
        )
    return EvalDeriv(basis).construct_array_mix(
        coord_type, points=points, orders=orders, deriv_type=deriv_type, mask=mask
    )
