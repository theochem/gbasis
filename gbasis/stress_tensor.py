"""Module for computing properties related to stress tensor."""
from gbasis.density import eval_density_laplacian, eval_deriv_density, eval_deriv_density_matrix
import numpy as np


# TODO: if this is symmetric, then we can symmeterize it instead of computing the entire array
# TODO: need to be tested against reference
def eval_stress_tensor(
    one_density_matrix, basis, coords, transform, alpha=1, beta=0, coord_type="spherical"
):
    r"""Return the stress tensor evaluated at the given coordinates.

    Stress tensor is defined here as:
    .. math::
        \boldsymbol{\sigma}_{ij}(\mathbf{r}_n | \alpha, \beta)
        =&
        -\frac{1}{2} \alpha
        \left(
            \frac{\partial^2}{\partial r_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
            + \frac{\partial^2}{\partial r_j \partial r'_i} \gamma(\mathbf{r}, \mathbf{r}')
        \right)_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\
        & +\frac{1}{2} (1 - \alpha)
        \left(
            \frac{\partial^2}{\partial r_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
            + \frac{\partial^2}{\partial r'_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
        \right)_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\
        & - \frac{1}{2} \delta_{ij} \beta
        \left.
            \nabla^2 \rho(\mathbf{r})
        \right_{\mathbf{r}=\mathbf{r}_n}\\
        =&
        - \alpha
        \left.
            \frac{\partial^2}{\partial r_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}
        + (1 - \alpha)
        \left.
            \frac{\partial^2}{\partial r_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}
        - \frac{1}{2} \delta_{ij} \beta
        \left.
            \nabla^2 \rho(\mathbf{r})
        \right_{\mathbf{r}=\mathbf{r}_n}\\

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix.
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from contractions in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left.
        Rows correspond to the linear combinationes (i.e. MO) and the columns correspond to the
        contractions (i.e. AO).
    alpha : {int, float}
        First parameter of the stress tensor.
        Default value is 1.
    beta : {int, float}
        Second parameter of the stress tensor.
        Default value is 0.
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    stress_tensor : np.ndarray(N, 3, 3)
        Stress tensor of the given density matrix at the given coordinates.

    Raises
    ------
    TypeError
        If `alpha` is not an integer or float.
        If `beta` is not an integer or float.

    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("`alpha` must be an integer or a float.")
    if not isinstance(beta, (int, float)):
        raise TypeError("`beta` must be an integer or a float.")
    output = np.zeros((3, 3, coords.shape[0]))
    for i, orders_two in enumerate(np.identity(3, dtype=int)):
        for j, orders_one in enumerate(np.identity(3, dtype=int)):
            if alpha != 0:
                output[i, j] -= alpha * eval_deriv_density_matrix(
                    orders_one,
                    orders_two,
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
            if alpha != 1:
                output[i, j] += (1 - alpha) * eval_deriv_density_matrix(
                    orders_one + orders_two,
                    np.array([0, 0, 0]),
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
            if i == j and beta != 0:
                output[i, j] -= (
                    0.5
                    * beta
                    * eval_density_laplacian(
                        one_density_matrix, basis, coords, transform, coord_type=coord_type
                    )
                )
    return np.transpose(output, (2, 1, 0))


# TODO: need to be tested against reference
def eval_ehrenfest_force(
    one_density_matrix, basis, coords, transform, alpha=1, beta=0, coord_type="spherical"
):
    r"""Return the Ehrenfest force.

    Ehrenfest force is the divergence of the stress tensor:
    .. math::

        F_{j}(\mathbf{r}_n | \alpha, \beta)
        =& \sum_i \frac{\partial}{\partial r_i} \boldsymbol{\sigma}_{ij}\\
        =&
        - \alpha
        \sum_i
        \left.
          \frac{\partial^3}{\partial r^2_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\
        &+ (1 - \alpha)
        \sum_i
        \left.
          \frac{\partial^3}{\partial r^2_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}
        + (1 - 2\alpha)
        \sum_i
        \left.
          \frac{\partial^3}{\partial r_i \partial r_j \partial r'_i} \gamma(\mathbf{r}, \mathbf{r})
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\
        &- \frac{1}{2} \beta
        \left.
          \left(
            \frac{\partial^3}{\partial r_j \partial x^2}
            + \frac{\partial^3}{\partial r_j \partial y^2}
            + \frac{\partial^3}{\partial r_j \partial z^2}
          \right)
          \rho(\mathbf{r}_n)
        \right|_{\mathbf{r}=\mathbf{r}_n}\\

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix.
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coords : np.ndarray(N, 3)
        Points in space where the contractions are evaluated.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from contractions in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left.
        Rows correspond to the linear combinationes (i.e. MO) and the columns correspond to the
        contractions (i.e. AO).
    alpha : {int, float}
        First parameter of the stress tensor.
        Default value is 1.
    beta : {int, float}
        Second parameter of the stress tensor.
        Default value is 0.
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    ehrenfest_force : np.ndarray(N, 3)
        Ehrenfest force of the given density matrix at the given coordinates.

    Raises
    ------
    TypeError
        If `alpha` is not an integer or float.
        If `beta` is not an integer or float.

    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("`alpha` must be an integer or a float.")
    if not isinstance(beta, (int, float)):
        raise TypeError("`beta` must be an integer or a float.")
    output = np.zeros((3, coords.shape[0]))
    for i, orders_two in enumerate(np.identity(3, dtype=int)):
        for orders_one in np.identity(3, dtype=int):
            if alpha != 0:
                output[i] -= alpha * eval_deriv_density_matrix(
                    2 * orders_one,
                    orders_two,
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
            if alpha != 1:
                output[i] += (1 - alpha) * eval_deriv_density_matrix(
                    2 * orders_one + orders_two,
                    np.array([0, 0, 0]),
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
            if alpha != 0.5:
                output[i] += (1 - 2 * alpha) * eval_deriv_density_matrix(
                    orders_one + orders_two,
                    orders_one,
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
            if beta != 0:
                output[i] -= (
                    0.5
                    * beta
                    * eval_deriv_density(
                        2 * orders_one + orders_two,
                        one_density_matrix,
                        basis,
                        coords,
                        transform,
                        coord_type=coord_type,
                    )
                )
    return output.T
