"""Module for computing the nuclear electron attraction."""
from gbasis.point_charge import point_charge_integral
import numpy as np


def nuclear_electron_attraction_integral(
    basis, nuclear_coords, nuclear_charges, transform=None, coord_type="spherical"
):
    """Return the nuclear electron attraction integrals of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each GeneralizedContractionShell instance.
        Default value is "spherical".

    Returns
    -------
    array : np.ndarray(K_cart, K_cart)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return np.sum(
        point_charge_integral(
            basis, nuclear_coords, nuclear_charges, transform=transform, coord_type=coord_type
        ),
        axis=2,
    )
