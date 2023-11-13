from autograd import grad
import autograd.numpy as np
from functools import partial

def eval_contractions(coords, center, angmom_comps, alphas, prim_coeffs, norm):
    """Return the evaluation of a Cartesian contraction.

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
    gauss = np.exp(-alphas * coords ** 2)

    zeroth_part = np.prod(coords ** angmom_comps * gauss, axis=(0, 2))
    # NOTE: `zeroth_part` now has axis 0 for primitives, axis 1 for angular momentum vector, and
    # axis 2 for coordinate

    norm = norm.T[:, :, np.newaxis]
    return np.tensordot(prim_coeffs, norm * zeroth_part, (0, 0)).flatten()

coords = np.random.rand(1, 1)
center = np.random.rand(1)
angmom_comps = np.random.randint(0, 2, (1, 1))
alphas = np.random.rand(1)
prim_coeffs = np.random.rand(1, 1)
norm = np.ones((1, 1))

output = eval_contractions(coords, center, angmom_comps, alphas, prim_coeffs, norm)
print(output)
eval_contruction_merged = partial(eval_contractions, 
                                  center=center,
                                  angmom_comps=angmom_comps,
                                  alphas=alphas,
                                  prim_coeffs=prim_coeffs,
                                  norm=norm)
eval_contractions_grad = grad(eval_contruction_merged)
print(eval_contractions_grad(coords))

# coords = coords.T[np.newaxis, np.newaxis, :, np.newaxis, :]
# center = center[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
# angmom_comps = angmom_comps.T[np.newaxis, np.newaxis, :, :, np.newaxis]
# alphas = alphas[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
# coords = coords - center
# gauss = np.exp(-alphas * coords ** 2)

# zeroth_part = np.prod(coords ** angmom_comps * gauss, axis=(0, 2))
# norm = norm.T[:, :, np.newaxis]
# np.tensordot(prim_coeffs, norm * zeroth_part, (0, 0))

# print()
# print(eval_contractions_grad(coords).shape)

