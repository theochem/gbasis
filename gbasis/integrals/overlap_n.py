"""
Arbitrary-order N-center Gaussian overlap integrals.

This module implements algorithms for computing overlap integrals
between Cartesian Gaussian basis functions located on different
centers. The implementation supports overlap integrals involving
an arbitrary number of Gaussian shells.

The evaluation follows the Gaussian product theorem and Hermite
Gaussian expansion techniques commonly used in quantum chemistry
integral engines.

References
----------
Helgaker, T., Jørgensen, P., Olsen, J.
Molecular Electronic-Structure Theory, Wiley (2000).

Szabo, A., Ostlund, N. S.
Modern Quantum Chemistry, Dover (1996).

Obara, S., Saika, A. (1986)
Efficient recursive computation of molecular integrals over
Cartesian Gaussian functions.
Journal of Chemical Physics.
"""

import numpy as np
from scipy.sparse import coo_matrix
from itertools import product


class PrimitiveNEngine:

    # Gaussian collapse
    @staticmethod
    def collapse_gaussians(alphas, centers):
        """
        Collapse multiple primitive Gaussian functions into a single
        effective Gaussian using the Gaussian product theorem.

        Given primitive Gaussian functions

            exp(-α_i |r - A_i|²)

        their product can be written as

            exp(-α_tot |r - P|²) * exp(term2 - term1)

        where

            α_tot = Σ α_i
            P = (Σ α_i A_i) / α_tot

        This transformation simplifies the evaluation of Gaussian
        integrals by reducing multiple Gaussians into a single
        Gaussian centered at P.

        Parameters
        ----------
        alphas : array_like
            Exponents of the primitive Gaussian functions.

        centers : array_like
            Cartesian coordinates of the Gaussian centers.

        Returns
        -------
        alpha_tot : float
            Total Gaussian exponent.

        P : ndarray
            Product center of the collapsed Gaussian.

        prefactor : float
            Exponential prefactor arising from Gaussian collapse.
            Raises:
            ------
            ValueError:
            If the total Gaussian exponent is non-positive.
        """
        alphas = np.asarray(alphas, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)

        alpha_tot = np.sum(alphas)

        if alpha_tot <= 0.0:
            raise ValueError("Total Gaussian exponent must be positive.")

        P = np.sum(alphas[:, None] * centers, axis=0) / alpha_tot

        term1 = np.sum(alphas * np.sum(centers**2, axis=1))
        term2 = alpha_tot * np.dot(P, P)

        exponent = term2 - term1
        prefactor = np.exp(exponent)

        return alpha_tot, P, prefactor

    # Pure binomial Hermite shift

    @staticmethod
    def hermite_coefficients(l, PA):
        """
        Compute Hermite translation coefficients.

        This function expands a polynomial centered at A in terms of
        powers centered at the Gaussian product center P:

        .. math::

            (x - A)^l = \\sum_t E_t (x - P)^t

        These coefficients are used in the Hermite Gaussian formalism
        for evaluating molecular integrals.

        Parameters
        ----------
        l : int
            Angular momentum order.

        PA : float
            Distance between the product center P and the original
            Gaussian center A.

        Returns
        -------
        numpy.ndarray
            Array containing the Hermite expansion coefficients.
        """
        E = np.zeros(l + 1, dtype=np.float64)
        E[0] = 1.0

        for i in range(l):
            E_new = np.zeros(l + 1, dtype=np.float64)

            for t in range(i + 1):
                E_new[t] += PA * E[t]
                E_new[t + 1] += E[t]

            E = E_new

        return E

    # Gaussian moments

    @staticmethod
    def gaussian_moments(alpha, max_order):
        """
        Compute Gaussian moment integrals.

        Evaluates integrals of the form

        .. math::

            \\int_{-\\infty}^{\\infty} (x-P)^k e^{-\\alpha (x-P)^2} dx

        Only even moments are non-zero. Higher moments are computed
        recursively from the zeroth moment.

        Parameters
        ----------
        alpha : float
            Gaussian exponent.

        max_order : int
            Maximum moment order to compute.

        Returns
        -------
        numpy.ndarray
            Array containing Gaussian moments up to ``max_order``.
        """
        moments = np.zeros(max_order + 1, dtype=np.float64)

        # zeroth moment
        moments[0] = np.sqrt(np.pi / alpha)

        # only even moments survive
        for k in range(0, max_order - 1, 2):
            moments[k + 2] = (k + 1) / (2.0 * alpha) * moments[k]

        return moments

    # Full primitive N-center overlap
    @staticmethod
    def primitive_overlap(alphas, centers, angmoms):
        """
        Compute primitive N-center Gaussian overlap integral.

        This function evaluates the overlap integral between N primitive
        Gaussian basis functions using the Hermite Gaussian formalism.

        The primitive overlap integral is

        .. math::

            S = \\int \\prod_i \\phi_i(\\mathbf{r}) \\, d\\mathbf{r}

        where each primitive Gaussian basis function has the form

        .. math::

            \\phi(\\mathbf{r}) =
            (x-A_x)^l (y-A_y)^m (z-A_z)^n
            e^{-\\alpha |\\mathbf{r}-A|^2}

        Parameters
        ----------
        alphas : list
            Gaussian exponents.

        centers : list
            Cartesian coordinates of Gaussian centers.

        angmoms : list
            Angular momentum tuples ``(l, m, n)``.

        Returns
        -------
        float
            Value of the primitive N-center overlap integral.

        Notes
        -----
        The overlap integral factorizes into independent Cartesian
        components (x, y, z). Each component is evaluated using
        Gaussian moment integrals obtained from the Hermite expansion.
        """
        alpha_tot, P, prefactor = PrimitiveNEngine.collapse_gaussians(alphas, centers)

        result = prefactor

        # factorize into x, y, z
        # Evaluate overlap as product of x, y, z integrals

        for axis in range(3):

            # build total polynomial via convolution
            E_total = np.array([1.0], dtype=np.float64)

            for i in range(len(alphas)):
                l = angmoms[i][axis]
                PA = P[axis] - centers[i][axis]

                E = PrimitiveNEngine.hermite_coefficients(l, PA)

                E_total = np.convolve(E_total, E)

            moments = PrimitiveNEngine.gaussian_moments(alpha_tot, len(E_total) - 1)

            axis_integral = np.dot(E_total, moments[: len(E_total)])

            result *= axis_integral

        return result


# Screening Function
def is_n_shell_overlap_screened(shells, tol=1e-12):
    """
    Determine whether an N-center overlap integral can be skipped
    using a conservative exponential bound.

    This screening estimates an upper bound for the magnitude of
    the contracted overlap integral using Gaussian decay between
    shell centers. If the bound is below the specified tolerance,
    the integral is considered negligible.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]
        Shells involved in the overlap integral.

    tol : float
        Screening tolerance.

    Returns
    -------
    bool
        True if the integral is negligible and can be skipped.
    """

    alpha_mins = [np.min(shell.exps) for shell in shells]
    centers = [shell.coord for shell in shells]

    alpha_tot = sum(alpha_mins)

    if alpha_tot <= 0.0:
        return True

    # Exponential decay from Gaussian collapse
    decay_sum = 0.0
    N = len(shells)

    for i in range(N):
        for j in range(i + 1, N):
            Rij = centers[i] - centers[j]
            Rij2 = np.dot(Rij, Rij)
            decay_sum += alpha_mins[i] * alpha_mins[j] * Rij2

    D = decay_sum / alpha_tot

    # Contraction-level magnitude bound
    coeff_bound = 1.0
    norm_bound = 1.0

    for shell in shells:
        coeff_bound *= np.max(np.abs(shell.coeffs))
        norm_bound *= np.max(np.abs(shell.norm_prim_cart))

    volume_bound = (np.pi / alpha_tot) ** 1.5

    bound = coeff_bound * norm_bound * volume_bound * np.exp(-D)

    return bound < tol


def contracted_n_overlap(shells):
    """
    Compute contracted N-center overlap for a list of
    GeneralizedContractionShell objects.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]

    Returns
    -------
    np.ndarray
        N-dimensional array over segmented contractions
        and Cartesian angular components.

        Shape:
        (M1, L1, M2, L2, ..., MN, LN)
    """

    N = len(shells)

    # Build shape for final tensor
    shape = []
    for shell in shells:
        shape.append(shell.num_seg_cont)
        shape.append(shell.num_cart)

    result = np.zeros(shape, dtype=np.float64)

    # Primitive exponent index ranges
    prim_ranges = [range(len(shell.exps)) for shell in shells]

    # Segmented contraction indices
    seg_ranges = [range(shell.num_seg_cont) for shell in shells]

    # Cartesian angular component indices
    cart_ranges = [range(shell.num_cart) for shell in shells]

    for seg_indices in product(*seg_ranges):
        for cart_indices in product(*cart_ranges):

            total_value = 0.0
            for prim_indices in product(*prim_ranges):

                alphas = []
                centers = []
                angmoms = []
                coeff_prod = 1.0
                norm_prod = 1.0

                for i, shell in enumerate(shells):

                    p = prim_indices[i]
                    m = seg_indices[i]
                    c = cart_indices[i]

                    alpha = shell.exps[p]
                    coeff = shell.coeffs[
                        p, m
                    ]  # contraction coefficient for primitive p in segment m
                    norm = shell.norm_prim_cart[
                        c, p
                    ]  # normalization factor for primitive p and Cartesian component c
                    angmom = tuple(
                        shell.angmom_components_cart[c]
                    )  # Cartesian angular momentum (lx, ly, lz)

                    alphas.append(alpha)
                    centers.append(shell.coord)
                    angmoms.append(angmom)

                    coeff_prod *= coeff
                    norm_prod *= norm

                prim_val = PrimitiveNEngine.primitive_overlap(alphas, centers, angmoms)

                total_value += coeff_prod * norm_prod * prim_val

            index = []
            for i in range(N):
                index.append(seg_indices[i])
                index.append(cart_indices[i])

            result[tuple(index)] = total_value

    return result


def build_n_overlap_tensor(shells, tol=1e-12):
    """
    Construct sparse N-center overlap tensor.

    This function builds the full N-center overlap tensor over
    Gaussian basis functions. Each tensor element corresponds
    to the overlap integral between a combination of contracted
    Gaussian shells.

    To reduce computational cost, shell-level screening is
    applied before evaluating the contracted integrals.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]
        List of Gaussian shells participating in the overlap
        integral.

    tol : float
        Screening tolerance used to discard negligible
        contributions.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse tensor containing the N-center overlap integrals.
        The tensor is flattened so that each N-index integral
        maps to a single row index.
    """

    # Total AO dimension
    shell_sizes = [shell.num_seg_cont * shell.num_cart for shell in shells]

    total_ao = sum(shell_sizes)
    N = len(shells)

    data = []
    rows = []

    # compute AO offsets per shell
    offsets = []
    acc = 0
    for size in shell_sizes:
        offsets.append(acc)
        acc += size

    for shell_indices in product(range(len(shells)), repeat=N):

        shell_tuple = [shells[i] for i in shell_indices]

        # Screening
        if is_n_shell_overlap_screened(shell_tuple, tol=tol):
            continue

        block = contracted_n_overlap(shell_tuple)

        block_flat = block.reshape(-1)

        local_sizes = [shells[i].num_seg_cont * shells[i].num_cart for i in shell_indices]

        local_offsets = [offsets[i] for i in shell_indices]

        for local_idx, value in enumerate(block_flat):

            if abs(value) < tol:
                continue

            # convert local multi-index to global index
            multi = []
            tmp = local_idx

            for size in reversed(local_sizes):
                multi.append(tmp % size)
                tmp //= size

            multi = list(reversed(multi))

            global_index = 0
            for k in range(N):
                global_index = global_index * total_ao + local_offsets[k] + multi[k]

            rows.append(global_index)
            data.append(value)

    shape = (total_ao**N, 1)

    return coo_matrix((data, (rows, np.zeros(len(rows)))), shape=shape)


# Public API function (Week 3 deliverable)
def arbitrary_order_overlap(shells, tol=1e-12):
    """
    Compute arbitrary-order Gaussian overlap tensor.

    This is the main public API for evaluating N-center overlap
    integrals between Gaussian basis functions.

    Internally this function constructs the overlap tensor using
    ``build_n_overlap_tensor``.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]
        List of Gaussian shells defining the basis functions.

    tol : float
        Screening tolerance.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse representation of the N-center overlap tensor.
    """

    return build_n_overlap_tensor(shells, tol)
