"""Module for computing the nuclear electron attraction."""
from gbasis.point_charge import (
    point_charge_cartesian,
    point_charge_spherical,
    point_charge_spherical_lincomb,
)
import numpy as np


def nuclear_electron_attraction_cartesian(basis, nuclear_coords, nuclear_charges):
    """Return the nuclear electron attraction integrals of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return np.sum(point_charge_cartesian(basis, nuclear_coords, nuclear_charges), axis=2)


def nuclear_electron_attraction_spherical(basis, nuclear_coords, nuclear_charges):
    """Return the nuclear electron attraction integrals of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return np.sum(point_charge_spherical(basis, nuclear_coords, nuclear_charges), axis=2)


def nuclear_electron_attraction_spherical_lincomb(
    basis, transform, nuclear_coords, nuclear_charges
):
    """Return the nuclear electron attraction integrals of the LCAO's in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Array whose first and second indices are associated with the linear combinations of the
        contracted spherical Gaussians.
        First and second indices of the array correspond to the linear combination of contracted
        spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
        combinations.

    """
    return np.sum(
        point_charge_spherical_lincomb(basis, transform, nuclear_coords, nuclear_charges), axis=2
    )
