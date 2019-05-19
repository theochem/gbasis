"""Module for computing the nuclear electron attraction."""
from gbasis.point_charge import PointChargeIntegral
import numpy as np


class NuclearElectronAttraction(PointChargeIntegral):
    """Class for evaluating the nuclear electron attraction integral.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first and second indices of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    boys_func : np.ndarray(np.ndarray(M, K_b, K_a))
        Boys function used to evaluate the one-electron integral.
        `M` is the number of orders that will be evaluated. `K_a` and `K_b` are the number of
        primitives on the left and right side, respectively.
    construct_array_contraction(self, contraction) : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
        Return the nuclear electron attraction integrals for the given
        `ContractedCartesianGaussians` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the nuclear electron attraction integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the nuclear electron attraction integrals associated with spherical Gaussians
        (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the nuclear electron attraction integrals associated with linear combinations
        of spherical Gaussians (linear combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @classmethod
    def construct_array_contraction(
        cls, contractions_one, contractions_two, nuclear_coords, nuclear_charges
    ):
        """Return nuclear_electron_attraction potential for the given contractions.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        nuclear_coords : np.ndarray(N_nuc, 3)
            Coordinates of each atom.
        nuclear_charges : np.ndarray(N_nuc)
            Charges of each atom.

        Raises
        ------
        TypeError
            If `contractions_one` is not a ContractedCartesianGaussians instance.
            If `contractions_two` is not a ContractedCartesianGaussians instance.
            If `nuclear_coords` is not a two-dimensional numpy array of int/float.
            If `nuclear_charges` is not a one-dimensional numpy array of int/float.
        ValueError
            If `nuclear_coords` does not have 3 columns.
            If `nuclear_coords` and `nuclear_charges` do not have the same number of colunmns.

        """
        output = super().construct_array_contraction(
            contractions_one, contractions_two, nuclear_coords, -nuclear_charges
        )
        return np.sum(output, axis=4)


def nuclear_electron_attraction_gbasis_cartesian(basis, nuclear_coords, nuclear_charges):
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
    return NuclearElectronAttraction(basis).construct_array_cartesian(
        nuclear_coords=nuclear_coords, nuclear_charges=nuclear_charges
    )


def nuclear_electron_attraction_gbasis_spherical(basis, nuclear_coords, nuclear_charges):
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
    return NuclearElectronAttraction(basis).construct_array_spherical(
        nuclear_coords=nuclear_coords, nuclear_charges=nuclear_charges
    )


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
    return NuclearElectronAttraction(basis).construct_array_spherical_lincomb(
        transform, nuclear_coords=nuclear_coords, nuclear_charges=nuclear_charges
    )
