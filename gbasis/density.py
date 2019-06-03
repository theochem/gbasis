"""Density Evaluation."""
from gbasis.eval import evaluate_basis_lincomb
from gbasis.eval_deriv import evaluate_deriv_basis_lincomb
import numpy as np
from scipy.special import comb


def eval_density_using_evaluated_orbs(one_density_matrix, orb_eval):
    """Return the evaluation of the density given the evaluated orbitals.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix.
    orb_eval : np.ndarray(K_orb, N)
        Orbitals evaluated at different grid points.
        The set of orbitals must be the same basis set used to build the one-electron density
        matrix.

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at different grid points.

    Raises
    ------
    TypeError
        If orb_eval is not a 2-dimensional numpy array with dtype float.
        If density_matrix is not a 2-dimensional numpy array with dtype float.
    ValueError
        If one-electron density matrix is not square.
        If the number of columns (or rows) of the one-electron density matrix is not equal to the
        number of rows of the orbital evaluations.

    """
    # test that inputs have the correct shape and type
    if not (
        isinstance(one_density_matrix, np.ndarray)
        and one_density_matrix.ndim == 2
        and one_density_matrix.dtype == float
    ):
        raise TypeError(
            "One-electron density matrix must be a two-dimensional numpy array with dtype float."
        )
    if not (isinstance(orb_eval, np.ndarray) and orb_eval.ndim == 2 and orb_eval.dtype == float):
        raise TypeError(
            "Evaluation of orbitals must be a two-dimensional numpy array with dtype float."
        )
    if one_density_matrix.shape[0] != one_density_matrix.shape[1]:
        raise ValueError("One-electron density matrix must be a square matrix.")
    if not np.allclose(one_density_matrix, one_density_matrix.T):
        raise ValueError("One-electron density matrix must be symmetric.")
    if one_density_matrix.shape[0] != orb_eval.shape[0]:
        raise ValueError(
            "Number of rows (and columns) of the density matrix must be equal to the number of rows"
            " of the orbital evaluations."
        )

    density = one_density_matrix.dot(orb_eval)
    density *= orb_eval
    return np.sum(density, axis=0)


def eval_density(one_density_matrix, basis, coords, transform, coord_type="spherical"):
    """Return the density of the given transformed basis set at the given coordinates.

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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at different grid points.

    """
    orb_eval = evaluate_basis_lincomb(basis, coords, transform, coord_type=coord_type)
    return eval_density_using_evaluated_orbs(one_density_matrix, orb_eval)


def eval_deriv_density_matrix(
    orders_one, orders_two, one_density_matrix, basis, coords, transform, coord_type="spherical"
):
    r"""Return the derivative of the density matrix evaluated at the same point.

    ..math::

        \left.
        \frac{\partial^{p_x + p_y + p_z}}{\partial x_1^{p_x} \partial y_1^{p_y} \partial z_1^{p_z}}
        \frac{\partial^{q_x + q_y + q_z}}{\partial x_2^{q_x} \partial y_2^{q_y} \partial z_2^{q_z}}
        \gamma(\mathbf{r}_1, \mathbf{r}_2)
        \right|_{\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}_n} =
        \sum_{ij} \gamma_{ij}
        \left.
        \frac{\partial^{p_x + p_y + p_z}}{\partial x_1^{p_x} \partial y_1^{p_y} \partial z_1^{p_z}}
        \phi_i(\mathbf{r}_1)
        \right|_{\mathbf{r}_1 = \mathbf{r}_n}
        \left.
        \frac{\partial^{q_x + q_y + q_z}}{\partial x_2^{q_x} \partial y_2^{q_y} \partial z_2^{q_z}}
        \phi_j(\mathbf{r}_2)
        \right|_{\mathbf{r}_1 = \mathbf{r}_n}

    where :math:`\mathbf{r}_1` is the first coordinate, :math:`\mathbf{r}_2` is the second
    coordinate, and :math:`\mathbf{r}_n` is the coordinate at which the derivative is evaluated.

    Parameters
    ----------
    orders_one : np.ndarray(3,)
        Orders of the derivative for the first coordinate.
    orders_two : np.ndarray(3,)
        Orders of the derivative for the second coordinate.
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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    deriv_density_matrix : np.ndarray(N,)
        Derivative of the density matrix evaluated at the given coordinates.

    """
    deriv_orb_eval_one = evaluate_deriv_basis_lincomb(
        basis, coords, orders_one, transform, coord_type=coord_type
    )
    if np.array_equal(orders_one, orders_two):
        deriv_orb_eval_two = deriv_orb_eval_one
    else:
        deriv_orb_eval_two = evaluate_deriv_basis_lincomb(
            basis, coords, orders_two, transform, coord_type=coord_type
        )
    density = one_density_matrix.dot(deriv_orb_eval_two)
    density *= deriv_orb_eval_one
    density = np.sum(density, axis=0)
    return density


def eval_deriv_density(
    orders, one_density_matrix, basis, coords, transform, coord_type="spherical"
):
    """Return the derivative of density of the given transformed basis set at the given coordinates.

    Parameters
    ----------
    orders : np.ndarray(3,)
        Orders of the derivative.
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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    density_deriv : np.ndarray(N,)
        Derivative of the density evaluated at different grid points.

    """
    # pylint: disable=R0914
    total_l_x, total_l_y, total_l_z = orders

    output = np.zeros(coords.shape[0])
    for l_x in range(total_l_x // 2 + 1):
        # prevent double counting for the middle of the even total_l_x
        # e.g. If total_l_x == 4, then l_x is in [0, 1, 2, 3, 4]. Exploiting symmetry we only need
        # to loop over [0, 1, 2] because l_x in [0, 4] and l_x in [1, 3] give the same result.
        # However, l_x = 2 needs to avoid double counting.
        if total_l_x % 2 == 0 and l_x == total_l_x / 2:
            factor = 1
        else:
            factor = 2
        for l_y in range(total_l_y + 1):
            for l_z in range(total_l_z + 1):
                num_occurence = comb(total_l_x, l_x) * comb(total_l_y, l_y) * comb(total_l_z, l_z)
                orders_one = np.array([l_x, l_y, l_z])
                orders_two = orders - orders_one
                density = eval_deriv_density_matrix(
                    orders_one,
                    orders_two,
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
                output += factor * num_occurence * density
    return output


def eval_density_gradient(one_density_matrix, basis, coords, transform, coord_type="spherical"):
    """Return the gradient of the density evaluated at the given coordinates.

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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    density_gradient : np.ndarray(N, 3)
        Gradient of the density evaluated at the given coordinates.

    """
    return np.array(
        [
            eval_deriv_density(
                np.array([1, 0, 0]),
                one_density_matrix,
                basis,
                coords,
                transform,
                coord_type=coord_type,
            ),
            eval_deriv_density(
                np.array([0, 1, 0]),
                one_density_matrix,
                basis,
                coords,
                transform,
                coord_type=coord_type,
            ),
            eval_deriv_density(
                np.array([0, 0, 1]),
                one_density_matrix,
                basis,
                coords,
                transform,
                coord_type=coord_type,
            ),
        ]
    ).T


def eval_density_laplacian(one_density_matrix, basis, coords, transform, coord_type="spherical"):
    """Return the Laplacian of the density evaluated at the given coordinates.

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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    density_laplacian : np.ndarray(N)
        Laplacian of the density evaluated at the given coordinates.

    """
    output = eval_deriv_density(
        np.array([2, 0, 0]), one_density_matrix, basis, coords, transform, coord_type=coord_type
    )
    output += eval_deriv_density(
        np.array([0, 2, 0]), one_density_matrix, basis, coords, transform, coord_type=coord_type
    )
    output += eval_deriv_density(
        np.array([0, 0, 2]), one_density_matrix, basis, coords, transform, coord_type=coord_type
    )
    return output


def eval_density_hessian(one_density_matrix, basis, coords, transform, coord_type="spherical"):
    """Return the Hessian of the density evaluated at the given coordinates.

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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    density_hessian : np.ndarray(N, 3, 3)
        Hessian of the density evaluated at the given coordinates.
        First index corresponds to the coordinate. Second and third indices correspond to the
        dimension (x, y, z) with respect to which the density is derivatized.

    """
    return np.array(
        [
            [
                eval_deriv_density(
                    orders_one + orders_two,
                    one_density_matrix,
                    basis,
                    coords,
                    transform,
                    coord_type=coord_type,
                )
                for orders_one in np.identity(3, dtype=int)
            ]
            for orders_two in np.identity(3, dtype=int)
        ]
    ).T


def eval_posdef_kinetic_energy_density(
    one_density_matrix, basis, coords, transform, coord_type="spherical"
):
    r"""Return evaluations of positive definite kinetic energy density.

    ..math::

        t_+ (\mathbf{r}_n)
        &= \frac{1}{2} \left.
          \nabla_{\mathbf{r}} \cdot \nabla_{\mathbf{r}'} \gamma(\mathbf{r}, \mathbf{r}')
        \right|_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\
        &= \frac{1}{2} \left(
          \frac{\partial^2}{\partial x \partial x'} \gamma(\mathbf{r}, \mathbf{r}')
          + \frac{\partial^2}{\partial y \partial y'} \gamma(\mathbf{r}, \mathbf{r}')
          + \frac{\partial^2}{\partial z \partial z'} \gamma(\mathbf{r}, \mathbf{r}')
        \right)_{\mathbf{r} = \mathbf{r}' = \mathbf{r}_n}\\

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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    posdef_kindetic_energy_density : np.ndarray(N,)
        Positive definite kintic energy density of the given transformed basis set evaluated at the
        given coordinates.

    """
    output = np.zeros(coords.shape[0])
    for orders in np.identity(3, dtype=int):
        output += eval_deriv_density_matrix(
            orders, orders, one_density_matrix, basis, coords, transform, coord_type=coord_type
        )
    return 0.5 * output


# TODO: test against a reference
def eval_general_kinetic_energy_density(
    one_density_matrix, basis, coords, transform, alpha, coord_type="spherical"
):
    r"""Return evaluations of general form of the kinetic energy density.

    .. math::

        t_{\alpha} (\mathbf{r}_n) = t_+(\mathbf{r}_n) + \alpha \nabla^2 \rho(\mathbf{r}_n)

    where :math:`t_+` is the positive definite kinetic energy density.

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
    alpha : float
        Parameter of the general form of the kinetic energy density.
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    general_kindetic_energy_density : np.ndarray(N,)
        General kintic energy density of the given transformed basis set evaluated at the given
        coordinates.

    Raises
    ------
    TypeError
        If `alpha` is not an integer or a float.

    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("`alpha` must be an int or float.")

    general_kinetic_energy_density = eval_posdef_kinetic_energy_density(
        one_density_matrix, basis, coords, transform, coord_type=coord_type
    )
    if alpha != 0:
        general_kinetic_energy_density += alpha * eval_density_laplacian(
            one_density_matrix, basis, coords, transform, coord_type=coord_type
        )
    return general_kinetic_energy_density
