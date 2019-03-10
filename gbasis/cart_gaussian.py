"""Evaluation of contracted Cartesian Gaussians."""
import numpy as np
# from gbasis.contractions import ContractedCartesianGaussians

def eval_cart (grid_pt, norms, angnom, coord, charge, coeffs, exps):
    r"""Evaluate the contracted Cart. Gaussians at a certain pt. for a shell.

    Parameters
    ----------
    grid_pt : np.ndarray(3,)
        Coordinate of the grid point where the function is evaluated.
    norms : np.ndarray(L,)
        Normilzation constants of Cartesian Gaussian for a shell,
        L = (angnom + 1) * (angnom + 2) / 2
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
    cont_carts : np.ndarray(L,)
        Value of the contracted Cartesian Gaussians at a point for all the shell, where
        L = (angnom + 1) * (angnom + 2) / 2

    """
    # Dimension of the shell with total angular momentum angnom
    shell_dim = (angnom + 1) * (angnom + 2) / 2
    # Initialize vector of the polynomial part of the primitive Cartesian Gaussians
    poly_vec = np.zeros(shell_dim, dtype = float)
    # Initialize counter for the position for poly_vec
    vec_pos = 0
    # Find all possible combinations of angular momentum on x, y, and z
    for angnom_z in xrange(0, angnom + 1):
        for angnom_y in xrange(0, angnom - angnom_y + 1):
            angnom_x = angnom - angnom_z - angnom_y
            vec_pos += 1
            poly_vec[vec_pos] = (grid_pt[0] - coord[0])**angnom_x
            poly_vec[vec_pos] *= (grid_pt[1] - coord[1])**angnom_y
            poly_vec[vec_pos] *= (grid_pt[2] - coord[2])**angnom_z

    # Initialize vector of the exponential part of the primitive Cartesian Gaussians
    exp_vec = np.zeros(coeffs.size + 1, dtype = float)
    # Initialize counter for the position for exp_vec
    vec_pos = 0
    # Evaluate exponential part of primitive Cartesian Gaussians
    for primitive in xrange(0, coeffs.size + 1):
        dist = (grid_pt[0] - coord[0])**2
        dist += (grid_pt[1] - coord[1])**2
        dist += (grid_pt[2] - coord[2])**2
        dist *= exps[primitive]
        exp_vec[vec_pos] = np.exp(-dist)

    # Contraction (TODO : must make sure of how the normalization constant is vectorized)
    prim_matrix = np.outer(primitive, poly_vec.T)
    for shell_in_atom in xrange(0, shell_dim + 1):
       for primitive in xrange(0, coeffs.size + 1):
           norm_pos = shell_in_atom * shell_dim + primitive
           cart_eval[shell_in_atom] += norms[norm_pos] * poly_vec[shell_in_atom]
                                                       * exp_vec[primitive]

    return cart_eval
