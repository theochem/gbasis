"""Derivative of a Gaussian Contraction."""

import numpy as np
from scipy.special import comb, eval_hermite, perm


# TODO: in the case of generalized Cartesian contraction where multiple shells have the same sets of
# exponents but different sets of primitive coefficients, it will be helpful to vectorize the
# `prim_coeffs` also.
# FIXME: name is pretty bad
# TODO: vectorize for multiple orders? Caching instead?
def _eval_deriv_contractions(coords, orders, center, angmom_comps, alphas, prim_coeffs, norm):
    """Return the evaluation of the derivative of a Cartesian contraction.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Point in space where the derivative of the Gaussian primitive is evaluated.
        Coordinates must be given as a two dimensional array, even if only one point is given.
    orders : np.ndarray(3,)
        Orders of the derivative.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(L, 3)
        Components of the angular momentum, :math:`(a_x, a_y, a_z)`.
        Angular momentum components must be given as a two dimensional array, even if only one
        set of components is given.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.
    prim_coeffs : np.ndarray(K, M)
        Contraction coefficients of the primitives.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm : np.ndarray(L, K)
        Normalization constants for the primitives in each contraction.

    Returns
    -------
    derivative : np.ndarray(M, L, N)
        Evaluation of the derivative at each given coordinate.
        Dimension 0 corresponds to the contraction, with `M` as the number of given contractions.
        Dimension 1 corresponds to the angular momentum vector, ordered as in `angmom_comps`.
        Dimension 2 corresponds to the point at which the derivative is evaluated, ordered as in
        `coords`.

    Notes
    -----
    The input is not checked. This means that you must provide the parameters as they are specified
    in the docstring. They must all be `numpy` arrays with the **correct shape**.

    Pople style basis sets are not supported. If multiple angular momentum vectors (with different
    angular momentum) and multiple contraction coefficients are provided, it is **not assumed** that
    the angular momentum vector should be paired up with the contraction coefficients. In fact, each
    angular momentum vector will create multiple contractions according to the given coefficients.

    """
    # pylint: disable=R0914
    # NOTE: following convention will be used to organize the axis of the multidimensional arrays
    # axis 0 = index for term in hermite polynomial (size: min(K, n)) where n is the order in given
    # dimension
    # axis 1 = index for primitive (size: K)
    # axis 2 = index for dimension (x, y, z) of coordinate (size: 3)
    # axis 3 = index for angular momentum vector (size: L)
    # axis 4 = index for coordinate (out of a grid) (size: N)
    # adjust the axis
    coords = coords.T[np.newaxis, np.newaxis, :, np.newaxis, :]
    # NOTE: if `coord` is two dimensional (3, N), then coords has shape (1, 1, 3, 1, N). If it is
    # one dimensional (3,), then coords has shape (1, 1, 3, 1)
    # NOTE: `order` is still assumed to be a one dimensional
    center = center[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    angmom_comps = angmom_comps.T[np.newaxis, np.newaxis, :, :, np.newaxis]
    # NOTE: if `angmom_comps` is two-dimensional (3, L), has shape (1, 1, 3, L). If it is one
    # dimensional (3, ) then it has shape (1, 1, 3)
    alphas = alphas[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    # NOTE: `prim_coeffs` will be used as a 1D array

    # shift coordinates
    coords = coords - center
    # useful variables
    gauss = np.exp(-alphas * coords**2)

    # zeroth order (i.e. no derivatization)
    indices_noderiv = orders <= 0

    zero_coords = coords[:, :, indices_noderiv]
    zero_angmom_comps = angmom_comps[:, :, indices_noderiv]
    zero_gauss = gauss[:, :, indices_noderiv]

    zeroth_part = np.prod(zero_coords**zero_angmom_comps * zero_gauss, axis=(0, 2))
    # NOTE: `zeroth_part` now has axis 0 for primitives, axis 1 for angular momentum vector, and
    # axis 2 for coordinate

    deriv_part = 1
    nonzero_orders = orders[~indices_noderiv]
    nonzero_orders = nonzero_orders[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    # derivatization part
    if nonzero_orders.size != 0:
        # get nonzero arrays
        nonzero_coords = coords[:, :, ~indices_noderiv]
        nonzero_angmom_comps = angmom_comps[:, :, ~indices_noderiv]
        nonzero_gauss = gauss[:, :, ~indices_noderiv]
        # General approach: compute the whole coefficients, zero out the irrelevant parts
        # NOTE: The following step assumes that there is only one set (nx, ny, nz) of derivatization
        # orders i.e. we assume that only one axis (axis 2) of `nonzero_orders` has a dimension
        # greater than 1
        indices_herm = np.arange(np.max(nonzero_orders) + 1)[:, None, None, None, None]
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
            * (-(alphas**0.5)) ** indices_herm
            * nonzero_coords**indices_angmom
        )
        # zero out the appropriate terms
        indices_zero = np.where(indices_herm < np.maximum(0, nonzero_orders - nonzero_angmom_comps))
        coeffs[indices_zero[0], :, indices_zero[2], indices_zero[3]] = 0
        indices_zero = np.where(nonzero_orders < indices_herm)
        coeffs[indices_zero[0], :, indices_zero[2]] = 0
        # compute
        # TODO: I don't know if the scipy.special.eval_hermite uses some smart vectorizing/caching
        # to evaluate multiple orders at the same time. Creating/finding a better function for
        # evaluating the hermite polynomial at different orders (in sequence) may be nice in the
        # future.
        hermite = np.sum(coeffs * eval_hermite(indices_herm, alphas**0.5 * nonzero_coords), axis=0)
        hermite = np.prod(hermite, axis=1)

        # NOTE: `hermite` now has axis 0 for primitives, 1 for angular momentum vector, and axis 2
        # for coordinates
        deriv_part = np.prod(nonzero_gauss, axis=(0, 2)) * hermite

    norm = norm.T[:, :, np.newaxis]
    return np.tensordot(prim_coeffs, norm * zeroth_part * deriv_part, (0, 0))


def _eval_first_second_order_deriv_contractions(
    coords, orders, center, angmom_comps, alphas, prim_coeffs, norm
):
    """Return the evaluation of direct 1st and 2nd derivative orders of a Cartesian contraction.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Point in space where the derivative of the Gaussian primitive is evaluated.
        Coordinates must be given as a two dimensional array, even if only one point is given.
    orders : np.ndarray(3,)
        Orders of the derivative.
        All orders must be lower than 2.
        Negative orders are treated as zero orders.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(L, 3)
        Components of the angular momentum, :math:`(a_x, a_y, a_z)`.
        Angular momentum components must be given as a two dimensional array, even if only one
        set of components is given.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.
    prim_coeffs : np.ndarray(K, M)
        Contraction coefficients of the primitives.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm : np.ndarray(L, K)
        Normalization constants for the primitives in each contraction.

    Returns
    -------
    derivative : np.ndarray(M, L, N)
        Evaluation of the derivative at each given coordinate.
        Dimension 0 corresponds to the contraction, with `M` as the number of given contractions.
        Dimension 1 corresponds to the angular momentum vector, ordered as in `angmom_comps`.
        Dimension 2 corresponds to the point at which the derivative is evaluated, ordered as in
        `coords`.

    Notes
    -----
    The input is not checked. This means that you must provide the parameters as they are specified
    in the docstring. They must all be `numpy` arrays with the **correct shape**.

    Pople style basis sets are not supported. If multiple angular momentum vectors (with different
    angular momentum) and multiple contraction coefficients are provided, it is **not assumed** that
    the angular momentum vector should be paired up with the contraction coefficients. In fact, each
    angular momentum vector will create multiple contractions according to the given coefficients.

    """

    # Useful variables
    new_coords = coords.T - center[None, :].T
    gauss = np.exp(-alphas[:, None, None] * (new_coords**2))

    # Filters derivative orders
    indices_noderiv = orders <= 0
    indices_first_deriv = orders == 1
    indices_second_deriv = orders == 2

    # Zeroth order derivative
    raw_zeroth_coords = new_coords[indices_noderiv]
    zeroth_angmom_comps = angmom_comps.T[indices_noderiv]
    zeroth_gauss = gauss[:, indices_noderiv]
    zeroth_coords = raw_zeroth_coords[:, None, :] ** zeroth_angmom_comps[:, :, None]
    raw_zeroth_deriv = zeroth_coords[None, :, :, :] * zeroth_gauss[:, :, None, :]
    zeroth_deriv = np.prod(raw_zeroth_deriv, axis=1)

    # Initialize first and second derivative variables with shape = zeroth_deriv.shape
    first_deriv = np.ones(zeroth_deriv.shape)
    second_deriv = np.ones(zeroth_deriv.shape)

    # Calling 1st and 2nd derivatives functions for different combination of orders
    if indices_first_deriv.any():
        first_deriv = _first_derivative(
            new_coords, gauss, indices_first_deriv, angmom_comps, alphas
        )
        if indices_second_deriv.any():
            second_deriv = _second_derivative(
                new_coords, gauss, indices_second_deriv, angmom_comps, alphas
            )
    elif indices_second_deriv.any():
        second_deriv = _second_derivative(
            new_coords, gauss, indices_second_deriv, angmom_comps, alphas
        )
        if indices_first_deriv.any():
            first_deriv = _first_derivative(
                new_coords, gauss, indices_first_deriv, angmom_comps, alphas
            )
    # Combining all the derivatives
    norm = norm.T[:, :, np.newaxis]
    output = np.tensordot(prim_coeffs, norm * zeroth_deriv * first_deriv * second_deriv, (0, 0))

    return output


def _first_derivative(center_coords, gauss, indices_first_deriv, angmom_comps, alphas):
    r"""Help function for calculation of explicit first derivative order for contracted gaussian.

    Parameters
    ----------
    center_coords : np.ndarray(N, 3)
        Shifted coordinates center_coords = coords - center
    gauss : np.ndarray(K, 3, N)
        variable containing ..math::e^{-\alpha\left(x-X_{A}\right)^{2}}
    indices_first_deriv: boolean np.array(L, 3)
        array contaning boolean values corresponding to coordinates for first derivative
    angmom_comps : np.ndarray(L, 3)
        Components of the angular momentum, :math:`(a_x, a_y, a_z)`.
        Angular momentum components must be given as a two dimensional array, even if only one
        set of components is given.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.

    Returns
    -------
    first order derivative : np.ndarray(K, L, N)
        Evaluation of first derivative at each given coordinate.
        Dimension 0 corresponds to number of primitives.
        Dimension 1 corresponds to the angular momentum vector, ordered as in `angmom_comps`.
        Dimension 2 corresponds to the point at which the derivative is evaluated, ordered as in
        `coords`.

    Notes
    -----
    This is a helper function for _eval_first_second_order_deriv_contractions() and gives an
    intermediate output that needs to be further processed to match general organization in Gbasis

    """

    first_coords = center_coords[indices_first_deriv]
    first_gauss = gauss[:, indices_first_deriv]
    first_ang_comp = angmom_comps.T[indices_first_deriv]
    # Indices to filter for ang momentum at the end
    n_0_indices = first_ang_comp == 0

    power_part_1 = first_ang_comp - 1
    # NOTE: the negative indices must be turned into zeros because (x-Xa) terms are sometimes
    # zero (and negative power is undefined).
    power_part_1[power_part_1 < 0] = 0
    part1 = first_coords[:, None, :] ** power_part_1[:, :, None]
    part2 = (2 * alphas[:, None, None]) * (first_coords**2)
    part2 = first_ang_comp[None, :, :, None] - part2[:, :, None, :]
    # NOTE: Using an array of ones with same shape as first_ang_comp to power part2_zero_ang_mom
    # variable in order to get the same shape as part2. This is done in order to make easier
    # to filter at the end for the angular components corresponding to n=0
    array_ones = np.ones(first_ang_comp.shape)
    part2_n_0 = (
        -2 * alphas[:, None, None, None] * (first_coords[:, None, :] ** array_ones[:, :, None])
    )
    raw_first_deriv = part1 * part2
    # Substitute angular components n=0 with correct derivative
    raw_first_deriv[:, n_0_indices, :] = part2_n_0[:, n_0_indices, :]
    raw_first_deriv = raw_first_deriv * first_gauss[:, :, None, :]
    first_deriv = np.prod(raw_first_deriv, axis=1)

    return first_deriv


def _second_derivative(center_coords, gauss, indices_second_deriv, angmom_comps, alphas):
    r"""Help function for calculation of explicit second derivative order for contracted gaussian.

    Parameters
    ----------
    center_coords : np.ndarray(N, 3)
        Shifted coordinates center_coords = coords - center
    gauss : np.ndarray(K, 3, N)
        variable containing ..math::e^{-\alpha\left(x-X_{A}\right)^{2}}
    indices_second_deriv: boolean np.array(L, 3)
        array contaning boolean values corresponding to coordinates for second derivative
    angmom_comps : np.ndarray(L, 3)
        Components of the angular momentum, :math:`(a_x, a_y, a_z)`.
        Angular momentum components must be given as a two dimensional array, even if only one
        set of components is given.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.

    Returns
    -------
    first order derivative : np.ndarray(K, L, N)
        Evaluation of second derivative at each given coordinate.
        Dimension 0 corresponds to number of primitives.
        Dimension 1 corresponds to the angular momentum vector, ordered as in `angmom_comps`.
        Dimension 2 corresponds to the point at which the derivative is evaluated, ordered as in
        `coords`.

    Notes
    -----
    This is a helper function for _eval_first_second_order_deriv_contractions() and gives an
    intermediate output that needs to be further processed to match general organization in Gbasis

    """

    second_coords = center_coords[indices_second_deriv]
    second_gauss = gauss[:, indices_second_deriv]
    second_ang_comp = angmom_comps.T[indices_second_deriv]
    # NOTE: As for the first derivative, using an array of ones with shape=second_ang_comp.shape
    # to match the shape of different variables, corresponding to calculations of n=0 and n=1
    # second derivatives, to make easier at the end to combine them.
    array_ones = np.ones(second_ang_comp.shape)
    # Indices to filter for ang momentum at the end
    n_1_indices = second_ang_comp == 1
    n_2_indices = second_ang_comp >= 2

    # angular momentum == 0
    total_n_0 = ((4 * alphas[:, None, None] ** 2) * (second_coords**2)) - (
        2 * alphas[:, None, None]
    )
    raw_second_deriv = total_n_0[:, :, None, :] ** array_ones[None, :, :, None]
    # angular momentum == 1
    if any(second_ang_comp[0] == 1):
        total_n_1 = ((4 * alphas[:, None, None] ** 2) * (second_coords**3)) - (
            (6 * alphas[:, None, None]) * second_coords
        )
        total_n_1 = total_n_1[:, :, None, :] ** array_ones[None, :, :, None]
        # Substitute angular components n=1 with correct derivative
        raw_second_deriv[:, n_1_indices, :] = total_n_1[:, n_1_indices, :]
        # angular momentum >= 2
        if any(second_ang_comp[0] >= 2):
            # Calculating ..math:: \left(x-X_{A}\right)^{n-2}
            power_part_1 = second_ang_comp - 2
            # NOTE: the negative indices must be turned into zeros because (x-Xa)
            # terms are sometimes zeros (and negative power is undefined).
            power_part_1[power_part_1 < 0] = 0
            part1_n_2 = second_coords[:, None, :] ** (power_part_1[:, :, None])
            # Calculating
            # ..math:: 4 \alpha^{2}\left(x-X_{A}\right)^{4}-
            # \alpha(4 n+2)\left(x-X_{A}\right)^{2}+n(n-1)
            part2_1_n_2 = (4 * alphas[:, None, None] ** 2) * (second_coords**4)
            part2_2_n_2 = (
                alphas[:, None, None, None]
                * (4 * second_ang_comp[:, :, None] + 2)
                * second_coords[:, None, :] ** 2
            )
            part2_3_n_2 = second_ang_comp * (second_ang_comp - 1)
            part2_n_2 = part2_1_n_2[:, :, None, :] - part2_2_n_2 + part2_3_n_2[None, :, :, None]
            total_n_2 = part1_n_2[None, :, :, :] * part2_n_2
            # Substitute angular components n=2 with correct derivative
            raw_second_deriv[:, n_2_indices, :] = total_n_2[:, n_2_indices, :]
    raw_second_deriv = raw_second_deriv * second_gauss[:, :, None, :]
    second_deriv = np.prod(raw_second_deriv, axis=1)

    return second_deriv
