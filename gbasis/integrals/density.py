"""
Density API built on top of the arbitrary-order Gaussian overlap engine.

This module provides high-level routines for constructing matrices of
Gaussian overlap integrals between shells. These matrices serve as
building blocks in density-related calculations such as intracule and
extracule analysis.

The overlap integrals are evaluated using the arbitrary-order Gaussian
overlap engine implemented in ``arbitrary_order_overlap``.
"""

import numpy as np

from .overlap_n import arbitrary_order_overlap


def compute_intracule(shells):
    """
    Compute pairwise intracule overlap matrix between Gaussian shells.

    This function constructs a matrix whose elements correspond to
    two-center overlap integrals between Gaussian basis functions.
    Each matrix element represents the overlap integral

    .. math::

        S_{ij} = \\int \\phi_i(\\mathbf{r}) \\phi_j(\\mathbf{r}) \\, d\\mathbf{r}

    where :math:`\\phi_i` and :math:`\\phi_j` are Cartesian Gaussian basis
    functions centered on different atoms or centers.

    The overlap integrals are evaluated using the
    :func:`arbitrary_order_overlap` engine, which computes Gaussian
    overlap tensors of arbitrary order.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]
        List of Gaussian shells defining the basis functions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, n)`` containing pairwise Gaussian
        overlap integrals between shells.

    Notes
    -----
    The intracule matrix is constructed by evaluating pairwise
    two-center overlap integrals between Gaussian shells. Each
    integral is extracted from the sparse tensor returned by the
    arbitrary-order overlap engine.

    References
    ----------
    Helgaker, T., Jørgensen, P., Olsen, J.
    *Molecular Electronic-Structure Theory*, Wiley (2000).

    Szabo, A., Ostlund, N. S.
    *Modern Quantum Chemistry*, Dover (1996).
    """
    n = len(shells)

    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            tensor = arbitrary_order_overlap([shells[i], shells[j]])

            # Extract scalar overlap value from sparse tensor returned by
            # the arbitrary-order overlap engine.
            value = tensor.data[0] if tensor.nnz > 0 else 0.0

            result[i, j] = value

    return result


def compute_extracule(shells):
    """
    Compute pairwise extracule overlap matrix between Gaussian shells.

    This function constructs a matrix of pairwise overlap integrals
    between Gaussian shells using the arbitrary-order overlap engine.

    Each matrix element corresponds to the two-center Gaussian
    overlap integral

    .. math::

        S_{ij} = \\int \\phi_i(\\mathbf{r}) \\phi_j(\\mathbf{r}) \\, d\\mathbf{r}

    where :math:`\\phi_i` and :math:`\\phi_j` are Cartesian Gaussian
    basis functions.

    Parameters
    ----------
    shells : list[GeneralizedContractionShell]
        List of Gaussian shells defining the basis functions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, n)`` containing pairwise Gaussian
        overlap integrals between shells.

    Notes
    -----
    This function provides an API wrapper around the
    :func:`arbitrary_order_overlap` routine for density-related
    calculations involving Gaussian basis functions.
    This function currently constructs the same overlap matrix as
    :func:`compute_intracule`, but is provided as a separate API
    entry point for extracule-related density calculations.

    References
    ----------
    Helgaker, T., Jørgensen, P., Olsen, J.
    *Molecular Electronic-Structure Theory*, Wiley (2000).
    """
    n = len(shells)

    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # evaluate two-center Gaussian overlap tensor between shells i and j
            tensor = arbitrary_order_overlap([shells[i], shells[j]])

            # extract scalar overlap value from sparse tensor
            value = tensor.data[0] if tensor.nnz > 0 else 0.0

            result[i, j] = value

    return result
