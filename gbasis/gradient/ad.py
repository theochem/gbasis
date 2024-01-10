from autograd import grad
import autograd.numpy as np

# it looks like there is a plan to do:
# 1. rewrite the calculation of normalization constants (see if there is a dependency on the R_a in the normalization constant)
# 2. rewrite the calculation of contractions
# 3. use autograd to calculate the gradient of the contractions

def eval_contractions(coords, R_x, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm):
    """Return the evaluation of a Cartesian contraction. 
    Note that it was addapted from the evals/_deriv.py file in the gbasis package to match the autograd API.

    Parameters
    ----------
    coords : np.ndarray(3)
        Point in space where the derivative of the Gaussian primitive is evaluated.
        Coordinates must be given as a one(!) dimensional array.
    center : np.ndarray(3,)
        Center of the Gaussian primitive.
    angmom_comps : np.ndarray(1, 3)
        Components of the angular momentum, :math:`(a_x, a_y, a_z)`.
        Angular momentum components must be given as a two dimensional array, even if only one
        set of components is given.
    alphas : np.ndarray(K,)
        Values of the (square root of the) precisions of the primitives.
    prim_coeffs : np.ndarray(K, 1)
        Contraction coefficients of the primitives.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm : np.ndarray(1, K)
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
    
    coords = coords.reshape(1, 3) # this is done so the following code from the gbasis package works
    coords = coords.T[np.newaxis, np.newaxis, :, np.newaxis, :]
    # NOTE: if `coord` is two dimensional (3, N), then coords has shape (1, 1, 3, 1, N). If it is
    # one dimensional (3,), then coords has shape (1, 1, 3, 1)
    # NOTE: `order` is still assumed to be a one dimensional
    center = np.array([R_x, R_y, R_z])
    center = center[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    angmom_comps = angmom_comps.reshape(1, 3)
    angmom_comps = angmom_comps.T[np.newaxis, np.newaxis, :, :, np.newaxis] #this line
    # NOTE: if `angmom_comps` is two-dimensional (3, L), has shape (1, 1, 3, L). If it is one
    # dimensional (3, ) then it has shape (1, 1, 3)
    alphas = alphas[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    # NOTE: `prim_coeffs` will be used as a 1D array
    
    # shift coordinates
    coords = coords - center
    # useful variables
    gauss = np.exp(-alphas * coords ** 2)

    zeroth_part = np.prod(coords ** angmom_comps * gauss, axis=(0, 2))
    # NOTE: `zeroth_part` now has axis 0 for primitives, axis 1 for angular momentum vector, and
    # axis 2 for coordinate

    norm = norm.reshape(1, -1)
    norm = norm.T[:, :, np.newaxis]
    return np.tensordot(prim_coeffs, norm * zeroth_part, (0, 0)).flatten()

def _eval_nuc_deriv(coords, orders, center, angmom_comps, alphas, prim_coeffs, norm):
    """
    Return the derivative of a Cartesian contraction with respect to nuclear coordinate.

    FixMe: this funciton works correctly only when M=1, L=1. 

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Points in space where the derivative of the Gaussian primitive is evaluated.
        Coordinates must be given as a two dimensional array.
    orders : np.ndarray(3, )
        Orders of the derivative. 
        Has to be posive integer numbers greater than zero
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
    derivative : np.ndarray(N, 3)
        Evaluation of the derivative at each given coordinate.
    """
    R_x, R_y, R_z = center
    
    # evaluating derivatives along each axis of the atomic coordinate
    g_x = grad(eval_contractions, 1)
    g_y = grad(eval_contractions, 2)
    g_z = grad(eval_contractions, 3)

    # evaluating higher order derivatives
    order_x, order_y, order_z = orders-1 # subtracting 1 because first order derivative has been calculated
    for i in range(order_x):
        g_x = grad(g_x, 1)
    for j in range(order_y):
        g_y = grad(g_y, 2)
    for k in range(order_z):
        g_z = grad(g_z, 3)

    # vectorizing derivatives to evaluate them at each coordinate efficiently
    grad_x = np.apply_along_axis(g_x, 1, coords, R_x, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm)
    grad_y = np.apply_along_axis(g_y, 1, coords, R_x, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm)
    grad_z = np.apply_along_axis(g_z, 1, coords, R_x, R_y, R_z, angmom_comps, alphas, prim_coeffs, norm)    

    return np.vstack([grad_x, grad_y, grad_z]).T

def eval_nuc_deriv(coords, orders, center, angmom_comps, alphas, prim_coeffs, norms):
    """
    Return the derivative of a Cartesian contraction with respect to nuclear coordinate.
    
    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Points in space where the derivative of the Gaussian primitive is evaluated.
        Coordinates must be given as a two dimensional array.
    orders : np.ndarray(3, )
        Orders of the derivative. 
        Has to be posive integer numbers greater than zero
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
    norms : np.ndarray(L, K)
        Normalization constants for the primitives in each contraction.

    Returns
    -------
    derivative : np.ndarray(M, L, N, 3)
        Evaluation of the derivative at each given coordinate.
    """
    M, L, N = prim_coeffs.shape[1], angmom_comps.shape[0], coords.shape[0]
    output = np.zeros((M, L, N, 3))

    for m in range(M):
        prim_coeff = prim_coeffs[:, m]

        angular_output = np.zeros((L, N, 3))
        for l in range(L):
            angmom_comp = angmom_comps[l]
            norm = norms[l]      

            derivative = _eval_nuc_deriv(coords, orders, center, angmom_comp, alphas, prim_coeff, norm)
            print(derivative.shape)
            angular_output[l] = derivative
        output[m] = angular_output
    return output