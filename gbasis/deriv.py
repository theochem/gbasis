"""Derivative of a Gaussian Contraction."""
import numpy as np
from scipy.special import comb, perm


# NOTE: coord, order, center, angmom_comps, and exp don't need to be checked because we will
# retrieve these information from the ContractedCartesianGaussians instance.
# FIXME: dumb implementation. should be utilizing ContractedCartesianGaussians
def eval_deriv_prim(coord, orders, center, angmom_comps, alpha):
    """Return the evaluation of the derivative of a Gaussian primitive.

    Parameters
    ----------
    coord : np.ndarray(3,)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alpha : float
        Value of the exponential in the Guassian primitive.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    Raises
    ------
    TypeError
        If x is not an integer or float.
        If orders is not an integer.
    ValueError
        If orders is less than or equal to zero.

    """
    rel_coord = coord - center
    gauss = np.exp(-alpha * rel_coord ** 2)
    # separate zero orders from non-zero orders
    zero_rel_coord, zero_angmom_comps, zero_gauss = (
        rel_coord[orders <= 0],
        angmom_comps[orders <= 0],
        gauss[orders <= 0],
    )
    nonzero_rel_coord, nonzero_orders, nonzero_angmom_comps, nonzero_gauss = (
        rel_coord[orders > 0],
        orders[orders > 0],
        angmom_comps[orders > 0],
        gauss[orders > 0],
    )
    # zeroth order (i.e. no derivatization)
    zeroth_part = np.prod(zero_rel_coord ** zero_angmom_comps * zero_gauss)
    deriv_part = 1

    # derivatization part
    if nonzero_orders.size != 0:
        # option 1: computer derivative of each primitive separately as a list
        all_indices = [
            np.arange(min(order, angmom_comp) + 1)
            for order, angmom_comp in zip(nonzero_orders, nonzero_angmom_comps)
        ]
        all_coeffs = [
            np.hstack(
                [
                    comb(nonzero_orders[i], all_indices[i])
                    * perm(nonzero_angmom_comps[i], all_indices[i])
                    * (-alpha ** 0.5) ** (nonzero_orders[i] - all_indices[i])
                    * nonzero_rel_coord[i] ** (nonzero_angmom_comps[i] - all_indices[i]),
                    np.zeros(nonzero_orders[i] - min(nonzero_orders[i], nonzero_angmom_comps[i])),
                ]
            )[::-1]
            for i in range(nonzero_orders.size)
        ]
        hermite = np.prod(
            [
                np.polynomial.hermite.hermval(alpha ** 0.5 * coord, coeffs)
                for coeffs, coord in zip(all_coeffs, nonzero_rel_coord)
            ]
        )

        # # option 2: compute the whole coefficents, zero out the irrelevant
        # indices = np.arange(max(nonzero_orders) + 1)[:, np.newaxis]
        # # get indices that are used as powers of the appropriate terms in the derivative
        # # NOTE: the negative indices must be turned into zeros (even though they are turned into
        # # zeros later anyways) because these terms are sometimes zeros (and negative power is
        # # undefined).
        # indices_angmom = nonzero_angmom_comps - nonzero_orders + indices
        # indices_angmom[indices_angmom < 0] = 0
        # # get coefficients for all entries
        # coeffs = (
        #     comb(nonzero_orders, indices)
        #     * perm(nonzero_angmom_comps, nonzero_orders - indices)
        #     * (-alpha ** 0.5) ** indices
        #     * nonzero_rel_coord ** indices_angmom
        # )
        # # zero out the appropriate terms
        # coeffs[indices < np.maximum(0, nonzero_orders - nonzero_angmom_comps)] = 0
        # coeffs[nonzero_orders < indices] = 0
        # # TODO: compare performance of the the two ways of computing hermite polynomial
        # # hermite = np.prod(
        # #     np.diag(np.polynomial.hermite.hermval(alpha ** 0.5 * nonzero_rel_coord, coeffs))
        # # )
        # hermite = np.prod(
        #     [
        #         np.polynomial.hermite.hermval(alpha ** 0.5 * nonzero_rel_coord[i], coeffs[:, i])
        #         for i in range(nonzero_rel_coord.size)
        #     ]
        # )

        # # option 3: zero coefficients, fill in the nonzero parts. This is similar to option 1 in
        # # the sense that we don't (cannot?) make full use of numpy, but we can evaluate the
        # # hermval polynomial at multiple orders
        # # NOTE: not too different from option 1
        # coeffs = np.zeros([max(nonzero_orders) + 1, nonzero_orders.size])
        # for i in range(nonzero_orders.size):
        #     indices = np.arange(max(nonzero_orders) + 1)[:, np.newaxis]
        #     indices = indices[
        #         np.logical_and(
        #             nonzero_orders[i] >= indices,
        #             indices >= max(0, nonzero_orders[i] - nonzero_angmom_comps[i])
        #         )
        #     ]
        #     coeffs[indices, i] = (
        #         comb(nonzero_orders[i], indices)
        #         * perm(nonzero_angmom_comps[i], nonzero_orders[i] - indices)
        #         * (-alpha ** 0.5) ** indices
        #         * nonzero_rel_coord[i] ** (nonzero_angmom_comps[i] - nonzero_orders[i] + indices)
        #     )
        # hermite = np.prod(
        #     np.diag(np.polynomial.hermite.hermval(alpha ** 0.5 * nonzero_rel_coord, coeffs))
        # )

        deriv_part = np.prod(nonzero_gauss) * hermite
    return zeroth_part * deriv_part


def eval_prim(coord, center, angmom_comps, alpha):
    """Return the evaluation of a Gaussian primitive.

    Parameters
    ----------
    coord : np.ndarray(3,)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alpha : float
        Value of the exponential in the Guassian primitive.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    Raises
    ------
    TypeError
        If x is not an integer or float.
        If orders is not an integer.
    ValueError
        If orders is less than or equal to zero.

    """
    return eval_deriv_prim(coord, np.zeros(angmom_comps.shape), center, angmom_comps, alpha)


def eval_deriv_contraction(coord, orders, center, angmom_comps, alphas, prim_coeffs):
    """Return the evaluation of the derivative of a Cartesian contraction.

    Parameters
    ----------
    coord : np.ndarray(3, N)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.
    prim_coeffs : np.ndarray(K,)
        Contraction coefficients of the primitives.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    """
    # pylint: disable=R0914
    # support primitive evaluation
    if isinstance(alphas, (int, float)):
        alphas = np.array([alphas])
    if isinstance(prim_coeffs, (int, float)):
        prim_coeffs = np.array([prim_coeffs])

    # NOTE: following convention will be used to organize the axis of the multidimensional arrays
    # axis 0 = index for term in hermite polynomial (size: min(K, n)) where n is the order in given
    # dimension
    # axis 1 = index for primitive (size: K)
    # axis 2 = index for dimension (x, y, z) of coordinate (size: 3)
    # axis 3 = index for coordinate (out of a grid) (size: N)
    # adjust the axis
    coord = coord[np.newaxis, np.newaxis]
    # NOTE: `order` is still assumed to be a one dimensional
    center = center[np.newaxis, np.newaxis, :]
    angmom_comps = angmom_comps[np.newaxis, np.newaxis, :]
    alphas = alphas[np.newaxis, :, np.newaxis]
    # NOTE: `prim_coeffs` will still be used as a 1D array

    # useful variables
    rel_coord = coord - center
    gauss = np.exp(-alphas * rel_coord ** 2)

    # zeroth order (i.e. no derivatization)
    indices_noderiv = orders <= 0
    zero_rel_coord, zero_angmom_comps, zero_gauss = (
        rel_coord[:, :, indices_noderiv],
        angmom_comps[:, :, indices_noderiv],
        gauss[:, :, indices_noderiv],
    )
    zeroth_part = np.prod(zero_rel_coord ** zero_angmom_comps * zero_gauss, axis=(0, 2))
    # NOTE: `zeroth_part` now has axis 0 for primitives and axis 1 for coordinates

    deriv_part = 1
    nonzero_rel_coord, nonzero_orders, nonzero_angmom_comps, nonzero_gauss = (
        rel_coord[:, :, ~indices_noderiv],
        orders[~indices_noderiv],
        angmom_comps[:, :, ~indices_noderiv],
        gauss[:, :, ~indices_noderiv],
    )
    nonzero_orders = nonzero_orders[np.newaxis, np.newaxis, :]

    # derivatization part
    if nonzero_orders.size != 0:
        # General approach: compute the whole coefficents, zero out the irrelevant parts
        # NOTE: The following step assumes that there is only one set (nx, ny, nz) of derivatization
        # orders i.e. we assume that only one axis (axis 2) of `nonzero_orders` has a dimension
        # greater than 1
        indices_herm = np.arange(np.max(nonzero_orders) + 1)[:, np.newaxis, np.newaxis]
        # get indices that are used as powers of the appropriate terms in the derivative
        # NOTE: the negative indices must be turned into zeros (even though they are turned into
        # zeros later anyways) because these terms are sometimes zeros (and negative power is
        # undefined).
        indices_angmom = nonzero_angmom_comps - nonzero_orders + indices_herm
        indices_angmom[indices_angmom < 0] = 0
        # get coefficients for all entries
        coeffs = (
            comb(nonzero_orders, indices_herm)
            * perm(nonzero_angmom_comps, nonzero_orders - indices_herm)
            * (-alphas ** 0.5) ** indices_herm
            * nonzero_rel_coord ** indices_angmom
        )
        # zero out the appropriate terms
        indices_zero = np.where(indices_herm < np.maximum(0, nonzero_orders - nonzero_angmom_comps))
        coeffs[indices_zero[0], :, indices_zero[2]] = 0
        indices_zero = np.where(nonzero_orders < indices_herm)
        coeffs[indices_zero[0], :, indices_zero[2]] = 0
        # compute
        # FIXME: I can't seem to vectorize the next part due to the API of
        # np.polynomial.hermite.hermval. The main problem is that the indices for the primitives and
        # the dimension must be constrained for the given `x` and `c`, otherwise the hermitian
        # polynomial is evaluated at many unnecessary points.
        hermite = np.prod(
            [
                [
                    np.polynomial.hermite.hermval(
                        alphas[:, i, 0] ** 0.5 * nonzero_rel_coord[:, 0, j], coeffs[:, i, j]
                    )
                    for i in range(alphas.shape[1])
                ]
                for j in range(nonzero_rel_coord.shape[2])
            ],
            # NOTE: for loop over the axis 1 (primitives) and 2 (dimension) moves it to axis 1 and
            # 0, respectively, while removing these indices from alphas and coeffs. hermval returns
            # an array of c.shape[1:] + x.shape.
            # Therefore, axis 0 is for index for dimension (x, y, z)
            #            axis 1 is the index for primitive
            #            axis 2 is the index for term in hermite polynomial
            #            axis 3 is the index for coordinates
            axis=(0, 2),
        )
        # NOTE: `hermite` now has axis 0 for primitives and axis 1 for coordinates
        deriv_part = np.prod(nonzero_gauss, axis=(0, 2)) * hermite

    return np.sum(prim_coeffs * zeroth_part * deriv_part, axis=0)


def eval_contraction(coord, center, angmom_comps, alphas, coeffs):
    """Return the evaluation of a Gaussian contraction.

    Parameters
    ----------
    coord : np.ndarray(3, N)
        Point in space where the derivative of the Gaussian primitive is evaluated.
    orders : np.ndarray(3,)
        Orders of the derivative.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(3,)
        Component of the angular momentum that corresponds to this dimension.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.
    prim_coeffs : np.ndarray(K,)
        Contraction coefficients of the primitives.

    Returns
    -------
    derivative : float
        Evaluation of the derivative.

    Returns
    -------
    contraction : float
        Evaluation of the contraction.

    """
    return eval_deriv_contraction(
        coord, np.zeros(center.shape), center, angmom_comps, alphas, coeffs
    )
