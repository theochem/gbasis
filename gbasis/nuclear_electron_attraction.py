"""Module for computing the nuclear electron attraction."""
from gbasis.point_charge import (
    point_charge_cartesian,
    point_charge_lincomb,
    point_charge_mix,
    point_charge_spherical,
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
        Array associated with the spherical contractions.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return np.sum(point_charge_spherical(basis, nuclear_coords, nuclear_charges), axis=2)


def nuclear_electron_attraction_mix(basis, nuclear_coords, nuclear_charges, coord_types):
    """Return the nuclear electron attraction integrals of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    coord_types : list/tuple of str
        Types of the coordinate system for each ContractedCartesianGaussians.
        Each entry must be one of "cartesian" or "spherical".

    Returns
    -------
    array : np.ndarray(K_cont, K_cont)
        Array associated with the contractions in the given coordinate systems.
        First and second indices of the array are associated with two contractions. `K_cont` is the
        total number of contractions within the instance.

    """
    return np.sum(point_charge_mix(basis, nuclear_coords, nuclear_charges, coord_types), axis=2)


def nuclear_electron_attraction_lincomb(
    basis, transform, nuclear_coords, nuclear_charges, coord_type="spherical"
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
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each ContractedCartesianGaussians instance.
        Default value is "spherical".

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Array whose first and second indices are associated with the linear combinations of the
        contractions.
        First and second indices of the array correspond to the linear combination of contractions.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """
    return np.sum(
        point_charge_lincomb(
            basis, transform, nuclear_coords, nuclear_charges, coord_type=coord_type
        ),
        axis=2,
    )
