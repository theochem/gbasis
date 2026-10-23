"""Density Evaluation."""

from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
import numpy as np
from scipy.special import comb


def evaluate_density_using_evaluated_orbs(one_density_matrix, orb_eval):
    """Return the evaluation of the density given the evaluated orbitals.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given orbitals.
    orb_eval : np.ndarray(K_orb, N)
        Orbitals evaluated at :math:`N` grid points.
        The set of orbitals must be the same basis set used to build the one-electron density
        matrix.

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at `N` grid points.

    Raises
    ------
    TypeError
        If `orb_eval` is not a 2-dimensional `numpy` array with `dtype` float.
        If `one_density_matrix` is not a 2-dimensional `numpy` array with `dtype` float.
    ValueError
        If `one_density_matrix` is not square.
        If the number of columns (or rows) of `one_density_matrix` is not equal to the number of
        rows of the orbital evaluations.

    """
    # test that inputs have the correct shape and type
    if not (
        isinstance(one_density_matrix, np.ndarray)
        and one_density_matrix.ndim == 2
        and one_density_matrix.dtype == float
    ):
        raise TypeError(
            "One-electron density matrix must be a two-dimensional `numpy` array with `dtype`"
            " float."
        )
    if not (isinstance(orb_eval, np.ndarray) and orb_eval.ndim == 2 and orb_eval.dtype == float):
        raise TypeError(
            "Evaluation of orbitals must be a two-dimensional `numpy` array with `dtype` float."
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


def evaluate_density(
    one_density_matrix,
    basis,
    points,
    transform=None,
    threshold=1.0e-8,
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the density of the given basis set at the given points.

    .. math::

        \rho(\mathbf{r}) = \sum_{ij} \gamma_{ij} \phi_i(\mathbf{r}) \phi_j(\mathbf{r})

    where :math:`\mathbf{r}` is the point at which the density is evaluated, :math:`\gamma_{ij}`
    is the one-electron density matrix, and :math:`\phi_i` is the :math:`i`-th basis function.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orbs, K_orbs)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    threshold : float, optional
        The absolute value below which negative density values are acceptable. Any negative density
        value with an absolute value smaller than this threshold will be set to zero.
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    density : np.ndarray(N,)
        Density evaluated at `N` grid points.

    """
    orb_eval = evaluate_basis(
        basis, points, transform=transform, screen_basis=screen_basis, tol_screen=tol_screen
    )
    output = evaluate_density_using_evaluated_orbs(one_density_matrix, orb_eval)
    # Fix #117: check magnitude of small negative density values, then use clip to remove them
    min_output = np.min(output)
    if min_output < 0.0 and abs(min_output) > threshold:
        raise ValueError(f"Found negative density <= {-threshold}, got {min_output}.")
    return output.clip(min=0.0)


def evaluate_deriv_reduced_density_matrix(
    orders_one,
    orders_two,
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the derivative of the first-order reduced density matrix at the given points.

    .. math::

        &\left.
        \frac{\partial^{p_x + p_y + p_z}}{\partial x_1^{p_x} \partial y_1^{p_y} \partial z_1^{p_z}}
        \frac{\partial^{q_x + q_y + q_z}}{\partial x_2^{q_x} \partial y_2^{q_y} \partial z_2^{q_z}}
        \gamma(\mathbf{r}_1, \mathbf{r}_2)
        \right|_{\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}}\\\\
        &=
        \sum_{ij} \gamma_{ij}
        \left.
        \frac{\partial^{p_x + p_y + p_z}}{\partial x_1^{p_x} \partial y_1^{p_y} \partial z_1^{p_z}}
        \phi_i(\mathbf{r}_1)
        \right|_{\mathbf{r}_1 = \mathbf{r}}
        \left.
        \frac{\partial^{q_x + q_y + q_z}}{\partial x_2^{q_x} \partial y_2^{q_y} \partial z_2^{q_z}}
        \phi_j(\mathbf{r}_2)
        \right|_{\mathbf{r}_2 = \mathbf{r}}

    where :math:`\mathbf{r}_1` is the first point, :math:`\mathbf{r}_2` is the second point, and
    :math:`\mathbf{r}` is the point at which the derivative is evaluated.

    Parameters
    ----------
    orders_one : np.ndarray(3,)
        Orders of the derivative for the first point, :math:`mathbf{r}_1`.
    orders_two : np.ndarray(3,)
        Orders of the derivative for the second point, :math:`mathbf{r}_1`.
    one_density_matrix : np.ndarray(K_orbs, K_orbs)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    deriv_reduced_density_matrix : np.ndarray(N,)
        Derivative of the first-order reduced density matrix evaluated at `N` grid points.

    """
    deriv_orb_eval_one = evaluate_deriv_basis(
        basis,
        points,
        orders_one,
        transform=transform,
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    if np.array_equal(orders_one, orders_two):
        deriv_orb_eval_two = deriv_orb_eval_one
    else:
        deriv_orb_eval_two = evaluate_deriv_basis(
            basis,
            points,
            orders_two,
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
    # density = one_density_matrix.dot(deriv_orb_eval_two)
    # density *= deriv_orb_eval_one
    # density = np.sum(density, axis=0)
    # return density
    return np.einsum("ij,jk,ik->k", one_density_matrix, deriv_orb_eval_two, deriv_orb_eval_one)


def evaluate_deriv_density(
    orders,
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the derivative of density of the given transformed basis set at the given points.

    .. math::

            &\frac{\partial^{L_x + L_y + L_z}}{\partial x^{L_x} \partial y^{L_y} \partial z^{L_z}}
            \rho(\mathbf{r})\\\\
            &=
            \sum_{l_x=0}^{L_x} \sum_{l_y=0}^{L_y} \sum_{l_z=0}^{L_z}
            \binom{L_x}{l_x} \binom{L_y}{l_y} \binom{L_z}{l_z}
            \sum_{ij} \gamma_{ij}
            \frac{\partial^{l_x + l_y + l_z} \rho(\mathbf{r})}{\partial x^{l_x} \partial y^{l_y} \partial z^{l_z}}
            \frac{
                \partial^{L_x + L_y + L_z - l_x - l_y - l_z} \rho(\mathbf{r})
            }{
                \partial x^{L_x - l_x} \partial y^{L_y - l_y} \partial z^{L_z - l_z}
            }

    where :math:`L_x, L_y, L_z` are the orders of the derivative relative to the :math:`x, y, \text{and} z` components,
    respectively.

    Parameters
    ----------
    orders : np.ndarray(3,)
        Orders of the derivative.
    one_density_matrix : np.ndarray(K_orbs, K_orbs)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    density_deriv : np.ndarray(N,)
        Derivative of the density evaluated at `N` grid points.

    """
    # pylint: disable=R0914
    total_l_x, total_l_y, total_l_z = orders

    output = np.zeros(points.shape[0])
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
                if any(orders_one > 2) or any(orders_two > 2):
                    density = evaluate_deriv_reduced_density_matrix(
                        orders_one,
                        orders_two,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                        deriv_type="general",
                        screen_basis=screen_basis,
                        tol_screen=tol_screen,
                    )
                else:
                    density = evaluate_deriv_reduced_density_matrix(
                        orders_one,
                        orders_two,
                        one_density_matrix,
                        basis,
                        points,
                        transform=transform,
                        deriv_type=deriv_type,
                        screen_basis=screen_basis,
                        tol_screen=tol_screen,
                    )
                output += factor * num_occurence * density
    return output


def evaluate_density_gradient(
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the gradient of the density evaluated at the given points.

    .. math::

            \nabla \rho(\mathbf{r})
            =
            \begin{bmatrix}
            \frac{\partial}{\partial x} \rho(\mathbf{r})\\\\
            \frac{\partial}{\partial y} \rho(\mathbf{r})\\\\
            \frac{\partial}{\partial z} \rho(\mathbf{r})
            \end{bmatrix}      

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    density_gradient : np.ndarray(N, 3)
        Gradient of the density evaluated at `N` grid points.

    """
    orders_one = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    output = np.zeros((3, len(points)))
    # Evaluation of generalized contraction shell for zeroth order = 0,0,0
    zeroth_deriv = evaluate_deriv_basis(
        basis,
        points,
        np.array([0, 0, 0]),
        transform=transform,
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )

    # Evaluation of generalized contraction shell for each partial derivative
    for ind, orders in enumerate(orders_one):
        deriv_comp = evaluate_deriv_basis(
            basis,
            points,
            orders,
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        # output[ind] = 2*(np.einsum('ij,ik,jk -> k',one_density_matrix, zeroth_deriv, deriv_comp))
        density = one_density_matrix.dot(zeroth_deriv)
        density *= deriv_comp
        output[ind] = 2 * 1 * np.sum(density, axis=0)
    return output.T


def evaluate_density_laplacian(
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the Laplacian of the density evaluated at the given points.

    .. math::

            \nabla^2 \rho(\mathbf{r})
            =
            \frac{\partial^2}{\partial x^2} \rho(\mathbf{r})
            + \frac{\partial^2}{\partial y^2} \rho(\mathbf{r})
            + \frac{\partial^2}{\partial z^2} \rho(\mathbf{r})

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    density_laplacian : np.ndarray(N)
        Laplacian of the density evaluated at `N` grid points.

    """
    orders_one_second = np.array(([2, 0, 0], [0, 2, 0], [0, 0, 2]))
    orders_one_first = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    orders_two = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    output = np.zeros(points.shape[0])
    # Evaluation of generalized contraction shell for zeroth order = 0,0,0
    zeroth_deriv = evaluate_deriv_basis(
        basis,
        points,
        np.array([0, 0, 0]),
        transform=transform,
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )

    # Evaluation of generalized contraction shell for each partial derivative
    for orders in orders_one_second:
        deriv_one = evaluate_deriv_basis(
            basis,
            points,
            orders,
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )

        density = one_density_matrix.dot(zeroth_deriv)
        density *= deriv_one
        output += 2 * 1 * np.sum(density, axis=0)

    for orders in zip(orders_one_first, orders_two):
        deriv_one = evaluate_deriv_basis(
            basis,
            points,
            orders[0],
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )

        deriv_two = evaluate_deriv_basis(
            basis,
            points,
            orders[1],
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        # output[ind] = 2*(np.einsum('ij,ik,jk -> k',one_density_matrix, zeroth_deriv, deriv_comp))
        density = one_density_matrix.dot(deriv_two)
        density *= deriv_one
        output += 2 * 1 * np.sum(density, axis=0)

    return output


def evaluate_density_hessian(
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return the Hessian of the density evaluated at the given points.

    .. math::

        H[\rho(\mathbf{r})]
        =
        \begin{bmatrix}
            \frac{\partial^2}{\partial x^2} \rho(\mathbf{r}) &
            \frac{\partial^2}{\partial x \partial y} \rho(\mathbf{r}) &
            \frac{\partial^2}{\partial x \partial z} \rho(\mathbf{r})\\\\
            \frac{\partial^2}{\partial y \partial x} \rho(\mathbf{r}) &
            \frac{\partial^2}{\partial y^2} \rho(\mathbf{r})&
            \frac{\partial^2}{\partial y \partial z} \rho(\mathbf{r})\\\\
            \frac{\partial^2}{\partial z \partial x} \rho(\mathbf{r}) &
            \frac{\partial^2}{\partial z \partial y} \rho(\mathbf{r})&
            \frac{\partial^2}{\partial z^2} \rho(\mathbf{r})\\
        \end{bmatrix}

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    density_hessian : np.ndarray(N, 3, 3)
        Hessian of the density evaluated at `N` grid points.
        Dimension 0 corresponds to the point, ordered as in `points`.
        Dimensions 1, 2 correspond to the dimensions `(x, y, z)` in which the derivative of density
        was calculated.

    """
    # Orders combined with zeroth derivative
    orders_one_zeroth = np.array(
        (
            [[2, 0, 0], [1, 1, 0], [1, 0, 1]],
            [[1, 1, 0], [0, 2, 0], [0, 1, 1]],
            [[1, 0, 1], [0, 1, 1], [0, 0, 2]],
        )
    )

    # Pairs of first order derivatives
    orders_one_two = np.array(
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
    )

    # Evaluation of generalized contraction shell for zeroth order = 0,0,0
    zeroth_deriv = evaluate_deriv_basis(
        basis,
        points,
        np.array([0, 0, 0]),
        transform=transform,
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )

    # Arrays for derivative
    zeroth_arr = np.full((3, 3, one_density_matrix.shape[0], points.shape[0]), zeroth_deriv)
    one_zeroth_arr = np.zeros((3, 3, one_density_matrix.shape[0], points.shape[0]))
    one_two_arr_1 = np.zeros((3, 3, one_density_matrix.shape[0], points.shape[0]))
    one_two_arr_2 = np.zeros((3, 3, one_density_matrix.shape[0], points.shape[0]))

    for i in range(3):
        for j in range(i + 1):
            # for j in range(3):
            one_zeroth_arr[j][i] = evaluate_deriv_basis(
                basis,
                points,
                orders_one_zeroth[j][i],
                transform=transform,
                deriv_type=deriv_type,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            )
            one_two_arr_1[j][i] = evaluate_deriv_basis(
                basis,
                points,
                orders_one_two[j][j],
                transform=transform,
                deriv_type=deriv_type,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            )
            one_two_arr_2[j][i] = evaluate_deriv_basis(
                basis,
                points,
                orders_one_two[j][i],
                transform=transform,
                deriv_type=deriv_type,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            )

    # double orders-zeroth density
    raw_density_1 = np.tensordot(one_zeroth_arr, one_density_matrix, (2, 1))
    density_1 = np.einsum("ijkm,ijmk -> ijkm", zeroth_arr, raw_density_1)

    # one_two density
    raw_density_2 = np.tensordot(one_two_arr_2, one_density_matrix, (2, 1))
    density_2 = np.einsum("ijkm,ijmk -> ijkm", one_two_arr_1, raw_density_2)

    # factors and sum over basis functions
    output = 2 * 1 * np.sum(density_1, axis=2)
    output += 2 * 1 * np.sum(density_2, axis=2)

    # copying lower matrix to upper matrix
    upp = np.swapaxes(output, 0, 1)
    upp = np.triu(upp.T, 1)
    return output.T + upp


def evaluate_posdef_kinetic_energy_density(
    one_density_matrix,
    basis,
    points,
    transform=None,
    deriv_type="general",
    threshold=1.0e-8,
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return evaluations of positive definite kinetic energy density at the given points.

    .. math::

        \begin{split}
        t_+ (\mathbf{r})
        &= \frac{1}{2} \left.
          \nabla_{\mathbf{r}_1} \cdot \nabla_{\mathbf{r}_2} \gamma(\mathbf{r}_1, \mathbf{r}_2)
        \right|_{\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}}\\
        &= \frac{1}{2} \left(
          \frac{\partial^2}{\partial x_1 \partial x_2} \gamma(\mathbf{r}_1, \mathbf{r}_2)
          + \frac{\partial^2}{\partial y_1 \partial y_2} \gamma(\mathbf{r}_1, \mathbf{r}_2)
          + \frac{\partial^2}{\partial z_1 \partial z_2} \gamma(\mathbf{r}_1, \mathbf{r}_2)
        \right)_{\mathbf{r}_1 = \mathbf{r}_2 = \mathbf{r}}\\
        \end{split}

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    transform : np.ndarray(K_orbs, K_cont), optional
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
    deriv_type : "general" or "direct", optional
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    threshold : float, optional
        The absolute value below which negative density values are acceptable. Any negative density
        value with an absolute value smaller than this threshold will be set to zero.
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    posdef_kinetic_energy_density : np.ndarray(N,)
        Positive definite kinetic energy density of the given transformed basis set evaluated at
        `N` grid points.

    """
    output = np.zeros(points.shape[0])
    for orders in np.identity(3, dtype=int):
        output += evaluate_deriv_reduced_density_matrix(
            orders,
            orders,
            one_density_matrix,
            basis,
            points,
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
    # Fix #117: check magnitude of small negative density values, then use clip to remove them
    min_output = np.min(output)
    if min_output < 0.0 and abs(min_output) > threshold:
        raise ValueError(f"Found negative density <= {-threshold}, got {min_output}.")
    return (0.5 * output).clip(min=0.0)


# TODO: test against a reference
def evaluate_general_kinetic_energy_density(
    one_density_matrix,
    basis,
    points,
    alpha,
    transform=None,
    deriv_type="general",
    screen_basis=False,
    tol_screen=1e-8,
):
    r"""Return evaluations of general form of the kinetic energy density at the given points.

    .. math::

        t_{\alpha} (\mathbf{r}) = t_+(\mathbf{r}) + \alpha \nabla^2 \rho(\mathbf{r})

    where :math:`t_+` is the positive definite kinetic energy density.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed to
        be expressed with respect to the transformed basis set.
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    alpha : float
        Parameter of the general form of the kinetic energy density.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    deriv_type : "general" or "direct"
        Specification of derivative of contraction function in _deriv.py. "general" makes reference
        to general implementation of any order derivative function (_eval_deriv_contractions())
        and "direct" makes reference to specific implementation of first and second order
        derivatives for generalized contraction (_eval_first_second_order_deriv_contractions()).
    screen_basis : bool, optional
        Whether to screen out points with negligible contributions. Default value is False.
    tol_screen : float
        Screening tolerance for excluding evaluations. Points with values below this tolerance
        will not be evaluated (they will be set to zero). Internal computed quantities that
        affect the results below this tolerance will also be ignored to speed up the
        evaluation. Default value is 1e-8.

    Returns
    -------
    general_kinetic_energy_density : np.ndarray(N,)
        General kinetic energy density of the given transformed basis set evaluated at `N`
        grid points.

    Raises
    ------
    TypeError
        If `alpha` is not an integer or a float.

    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("`alpha` must be an int or float.")

    general_kinetic_energy_density = evaluate_posdef_kinetic_energy_density(
        one_density_matrix,
        basis,
        points,
        transform=transform,
        deriv_type=deriv_type,
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    if alpha != 0:
        general_kinetic_energy_density += alpha * evaluate_density_laplacian(
            one_density_matrix,
            basis,
            points,
            transform=transform,
            deriv_type=deriv_type,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
    return general_kinetic_energy_density
