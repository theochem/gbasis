"""Evaluation of contracted Cartesian Gaussians."""
import numpy as np
# from gbasis.contractions import ContractedCartesianGaussians

def eval_cart (grid_pt, angnom, coord, charge, coeffs, exps):
    r"""Evaluate the contracted Cartesian Gaussian at a certain point.

    Parameters
    ----------
    grid_pt : np.ndarray(3,)
        Coordinate of the grid point where the function is evaluated.
    angnom : int 
        Angular momentum of the set of contractions.
    coord : np.ndarray(3,)
        Coordinate of the center of the Gaussian primitives.
    charge : float
        Charge at the center of the Gaussian primitives.
    coeffs : np.ndarray(K,)
        Contraction coefficients of the Gaussian primitives.
    exps : np.ndarray(K,)
        Exponents of the Gaussian primitives.

    Returns
    -------
    cart_gauss : np.ndarray(3,)
        Value of the Cartesian Gaussian function at a point.

    """
    for
