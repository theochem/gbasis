import numpy as np
from itertools import product


def three_gaussian_overlap_primitive(prims):
    alphas = np.array([p[0] for p in prims])
    centers = np.array([p[1] for p in prims])

    alpha_tot = np.sum(alphas)
    P = np.sum(alphas[:, None] * centers, axis=0) / alpha_tot

    term1 = np.sum(alphas * np.sum(centers**2, axis=1))
    term2 = alpha_tot * np.dot(P, P)

    prefactor = np.exp(-(term1 - term2))
    return prefactor * (np.pi / alpha_tot) ** 1.5


def three_overlap_tensor(basis):
    """
    basis: list of primitives [(alpha, center)]
    returns T[μ, ν, λ]
    """
    n = len(basis)
    T = np.zeros((n, n, n))

    for μ, ν, λ in product(range(n), repeat=3):
        T[μ, ν, λ] = three_gaussian_overlap_primitive(
            [basis[μ], basis[ν], basis[λ]]
        )

    return T
