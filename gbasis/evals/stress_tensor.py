"""Module for computing properties related to the stress tensor."""
from gbasis.evals.density import (
    evaluate_density_laplacian,
    evaluate_deriv_density,
    evaluate_deriv_reduced_density_matrix,
)
import numpy as np


# TODO: need to be tested against reference
def evaluate_stress_tensor(one_density_matrix, basis, points, alpha=1, beta=0, transform=None):
    r"""Return the stress tensor evaluated at the given coordinates.

    Stress tensor is defined here as:

    .. math::

            \boldsymbol{\sigma}_{ij}(\mathbf{r} | \alpha, \beta)
            =&
            -\frac{1}{2} \alpha
            \left(
                \frac{\partial^2}{\partial r_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
                + \frac{\partial^2}{\partial r_j \partial r'_i} \gamma(\mathbf{r}, \mathbf{r}')
            \right)_{\mathbf{r} = \mathbf{r}'}\\
            & +\frac{1}{2} (1 - \alpha)
            \left(
                \frac{\partial^2}{\partial r_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
                + \frac{\partial^2}{\partial r'_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
            \right)_{\mathbf{r} = \mathbf{r}'}\\
            & - \frac{1}{2} \delta_{ij} \beta
            \left.
                \nabla^2 \rho(\mathbf{r})
            \right)\\
            =&
            - \alpha
            \left.
                \frac{\partial^2}{\partial r_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
            \right|_{\mathbf{r} = \mathbf{r}'}
            + (1 - \alpha)
            \left.
                \frac{\partial^2}{\partial r_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
            \right|_{\mathbf{r} = \mathbf{r}'}
            - \frac{1}{2} \delta_{ij} \beta
            \left.
                \nabla^2 \rho(\mathbf{r})
            \right)

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis
        functions are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    alpha : {int, float}
        First parameter of the stress tensor.
        Default value is 1.
    beta : {int, float}
        Second parameter of the stress tensor.
        Default value is 0.

    Returns
    -------
    stress_tensor : np.ndarray(N, 3, 3)
        Stress tensor of the given density matrix evaluated at the given points.

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
    output = np.zeros((3, 3, points.shape[0]))
    for i, orders_two in enumerate(np.identity(3, dtype=int)):
        for j, orders_one in enumerate(np.identity(3, dtype=int)[i:]):
            j += i
            if alpha != 0:
                output[i, j] -= alpha * evaluate_deriv_reduced_density_matrix(
                    orders_one,
                    orders_two,
                    one_density_matrix,
                    basis,
                    points,
                    transform=transform,
                )
            if alpha != 1:
                output[i, j] += (1 - alpha) * evaluate_deriv_reduced_density_matrix(
                    orders_one + orders_two,
                    np.array([0, 0, 0]),
                    one_density_matrix,
                    basis,
                    points,
                    transform=transform,
                )
            if i == j and beta != 0:
                output[i, j] -= (
                    0.5
                    * beta
                    * evaluate_density_laplacian(
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                )
            output[j, i] = output[i, j]
    return np.transpose(output, (2, 1, 0))


# TODO: need to be tested against reference
def evaluate_ehrenfest_force(one_density_matrix, basis, points, alpha=1, beta=0, transform=None):
    r"""Return the Ehrenfest force.

    Ehrenfest force is the negative of the divergence of the stress tensor:

    .. math::

            F_{j}(\mathbf{r} | \alpha, \beta)
            =&- \sum_i \frac{\partial}{\partial r_i} \boldsymbol{\sigma}_{ij}\\
            =&
            \alpha
            \sum_i
            \left.
              \frac{\partial^3}{\partial r^2_i \partial r'_j} \gamma(\mathbf{r}, \mathbf{r}')
            \right|_{\mathbf{r} = \mathbf{r}'}\\
            &- (1 - \alpha)
            \sum_i
            \left.
              \frac{\partial^3}{\partial r^2_i \partial r_j} \gamma(\mathbf{r}, \mathbf{r})
            \right|_{\mathbf{r} = \mathbf{r}'}
            - (1 - 2\alpha)
            \sum_i
            \left.
              \frac{\partial^3}{\partial r_i \partial r_j \partial r'_i} \gamma(\mathbf{r}, \mathbf{r})
            \right|_{\mathbf{r} = \mathbf{r}'}\\
            &+ \frac{1}{2} \beta
            \left(
            \frac{\partial^3}{\partial r_j \partial x^2}
            + \frac{\partial^3}{\partial r_j \partial y^2}
            + \frac{\partial^3}{\partial r_j \partial z^2}
            \right)
            \rho(\mathbf{r})

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis
        functions are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    alpha : {int, float}
        First parameter of the stress tensor.
        Default value is 1.
    beta : {int, float}
        Second parameter of the stress tensor.
        Default value is 0.

    Returns
    -------
    ehrenfest_force : np.ndarray(N, 3)
        Ehrenfest force of the given density matrix evaluated at the given coordinates.

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
    output = np.zeros((3, points.shape[0]))
    for i, orders_two in enumerate(np.identity(3, dtype=int)):
        for orders_one in np.identity(3, dtype=int):
            if alpha != 0:
                output[i] += alpha * evaluate_deriv_reduced_density_matrix(
                    2 * orders_one,
                    orders_two,
                    one_density_matrix,
                    basis,
                    points,
                    transform=transform,
                )
            if alpha != 1:
                output[i] -= (1 - alpha) * evaluate_deriv_reduced_density_matrix(
                    2 * orders_one + orders_two,
                    np.array([0, 0, 0]),
                    one_density_matrix,
                    basis,
                    points,
                    transform=transform,
                )
            if alpha != 0.5:
                output[i] -= (1 - 2 * alpha) * evaluate_deriv_reduced_density_matrix(
                    orders_one + orders_two,
                    orders_one,
                    one_density_matrix,
                    basis,
                    points,
                    transform=transform,
                )
            if beta != 0:
                output[i] += (
                    0.5
                    * beta
                    * evaluate_deriv_density(
                        2 * orders_one + orders_two,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                )
    return output.T


# TODO: need to be tested against reference
def evaluate_ehrenfest_hessian(
    one_density_matrix,
    basis,
    points,
    alpha=1,
    beta=0,
    transform=None,
    symmetric=False,
):
    r"""Return the Ehrenfest Hessian.

    Ehrenfest Hessian is the gradient of the Ehrenfest force:

    .. math::

            H_{jk}(\mathbf{r} | \alpha, \beta)
            =&
            - \frac{\partial}{\partial r_k} F_j(\mathbf{r} | \alpha, \beta)\\
            =&
            \alpha
            \sum_i
            \left(
                \frac{\partial^4}{\partial r^2_i \partial r_k \partial r'_j}
                \gamma(\mathbf{r}, \mathbf{r}')
                +\frac{\partial^4}{\partial r^2_i \partial r'_j \partial r'_k}
                \gamma(\mathbf{r}, \mathbf{r}')
            \right)_{\mathbf{r} = \mathbf{r}'}\\
            &- (1 - \alpha)
            \sum_i
            \left(
                \frac{\partial^4}{\partial r^2_i \partial r_j \partial r_k}
                \gamma(\mathbf{r}, \mathbf{r})
                + \frac{\partial^4}{\partial r^2_i \partial r_j \partial r'_k}
                \gamma(\mathbf{r}, \mathbf{r})
            \right)_{\mathbf{r} = \mathbf{r}'}\\
            &- (1 - 2\alpha)
            \sum_i
            \left(
                \frac{\partial^4}{\partial r_i \partial r_j \partial r_k \partial r'_i}
                \gamma(\mathbf{r}, \mathbf{r})
                + \frac{\partial^4}{\partial r_i \partial r_j \partial r'_i \partial r'_k}
                \gamma(\mathbf{r}, \mathbf{r})
            \right)_{\mathbf{r} = \mathbf{r}'}\\
            &+ \frac{1}{2} \beta
                \left(
                    \frac{\partial^4}{\partial r_j \partial r_k \partial x^2}
                    + \frac{\partial^4}{\partial r_j \partial r_k \partial y^2}
                    + \frac{\partial^4}{\partial r_j \partial r_k \partial z^2}
                \right)
                \rho(\mathbf{r})

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis
        functions are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    alpha : {int, float}
        First parameter of the stress tensor.
        Default value is 1.
    beta : {int, float}
        Second parameter of the stress tensor.
        Default value is 0.
    symmetric : {True, False}
        Flag for symmetrizing the Hessian.
        If True, then the Hessian is symmetrized by averaging it with its transpose.
        Default value is False.

    Returns
    -------
    ehrenfest_hessian : np.ndarray(N, 3, 3)
        Ehrenfest Hessian of the given density matrix evaluated at the given coordinates.
        Hessian is symmetrized if `symmetric` is True.

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
    output = np.zeros((3, 3, points.shape[0]))
    for i, orders_two in enumerate(np.identity(3, dtype=int)):
        for j, orders_three in enumerate(np.identity(3, dtype=int)):
            for orders_one in np.identity(3, dtype=int):
                if alpha != 0:
                    output[i, j] += alpha * evaluate_deriv_reduced_density_matrix(
                        2 * orders_one + orders_three,
                        orders_two,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                    output[i, j] += alpha * evaluate_deriv_reduced_density_matrix(
                        2 * orders_one,
                        orders_two + orders_three,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                if alpha != 1:
                    output[i, j] -= (1 - alpha) * evaluate_deriv_reduced_density_matrix(
                        2 * orders_one + orders_two + orders_three,
                        np.array([0, 0, 0]),
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                    output[i, j] -= (1 - alpha) * evaluate_deriv_reduced_density_matrix(
                        2 * orders_one + orders_two,
                        orders_three,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                if alpha != 0.5:
                    output[i, j] -= (1 - 2 * alpha) * evaluate_deriv_reduced_density_matrix(
                        orders_one + orders_two + orders_three,
                        orders_one,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                    output[i, j] -= (1 - 2 * alpha) * evaluate_deriv_reduced_density_matrix(
                        orders_one + orders_two,
                        orders_one + orders_three,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                    )
                if beta != 0:
                    output[i, j] += (
                        0.5
                        * beta
                        * evaluate_deriv_density(
                            2 * orders_one + orders_two + orders_three,
                            one_density_matrix,
                            basis,
                            points,
                            transform=transform,
                        )
                    )
    if symmetric:
        output += np.swapaxes(output, 0, 1)
        output /= 2
    return np.transpose(output, (2, 0, 1))
